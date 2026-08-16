import os, joblib, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from data import fight_observation

RICH = {
 'current_win_streak':'current_win_streak','current_lose_streak':'current_lose_streak',
 'wins':'wins','losses':'losses','rounds':'total_rounds_fought','title_bouts':'total_title_bouts',
 'ko':'win_by_ko_tko','subwin':'win_by_submission','sig':'avg_sig_str_landed',
 'sigpct':'avg_sig_str_pct','sub':'avg_sub_att','td':'avg_td_landed','tdpct':'avg_td_pct',
 'height':'height_cms','reach':'reach_cms','age':'age','rank':'match_weightclass_rank'
}

class FighterState:
 def __init__(self):
  self.elo=1500.; self.n=0; self.wins=0; self.last=[]; self.stats={'sig':[],'td':[],'sub':[],'kd':[]}; self.profile={}
 def snapshot(self):
  def avg(k): return float(np.mean(self.stats[k][-5:])) if self.stats[k] else 0.
  p={k:float(v) for k,v in self.profile.items() if v is not None and np.isfinite(v)}
  p.update({'elo':self.elo,'n':self.n,'winrate':self.wins/self.n if self.n else .5,'recent_win':np.mean(self.last[-5:]) if self.last else .5,'sig5':avg('sig'),'td5':avg('td'),'sub5':avg('sub'),'kd5':avg('kd')})
  return p
 def update(self,win,obs,row=None,prefix=None):
  self.n+=1; self.wins+=int(win); self.last.append(int(win)); self.last=self.last[-20:]
  for k in self.stats:
   self.stats[k].append(float(obs.get(k,0))); self.stats[k]=self.stats[k][-20:]
  if row is not None and prefix:
   for out,base in RICH.items():
    c=f'{prefix}_{base}'
    if c in row.index:
     try:
      v=float(row[c]);
      if np.isfinite(v): self.profile[out]=v
     except: pass

def state_for(states,name):
 if name not in states: states[name]=FighterState()
 return states[name]

def _get(p,k): return float(p.get(k,0.0))

def features(sa,sb):
 a,b=sa.snapshot(),sb.snapshot()
 # Differences only: preserves matchup symmetry and avoids using post-fight outcome fields.
 vals=[
  (_get(a,'elo')-_get(b,'elo'))/400,
  _get(a,'winrate')-_get(b,'winrate'), _get(a,'recent_win')-_get(b,'recent_win'),
  (_get(a,'sig5')-_get(b,'sig5'))/50, (_get(a,'td5')-_get(b,'td5'))/5, (_get(a,'sub5')-_get(b,'sub5'))/2,
  (_get(a,'kd5')-_get(b,'kd5'))/2, min(_get(a,'n'),20)/20-min(_get(b,'n'),20)/20,
 ]
 for k,s in [('current_win_streak',5),('current_lose_streak',5),('wins',20),('losses',20),('rounds',50),('title_bouts',5),('ko',10),('subwin',10),('sig',5),('sigpct',1),('sub',2),('td',2),('tdpct',1),('height',20),('reach',20),('age',10),('rank',10)]:
  vals.append((_get(a,k)-_get(b,k))/s)
 return np.nan_to_num(np.asarray(vals,float),nan=0.0,posinf=0.0,neginf=0.0)

def update_elo(sa,sb,y):
 expected=1/(1+10**((sb.elo-sa.elo)/400)); k=28 if min(sa.n,sb.n)>5 else 36
 sa.elo+=k*(y-expected); sb.elo+=k*((1-y)-(1-expected))

def make_xy(d):
 states={}; X=[]; y=[]
 for _,r in d.iterrows():
  sa,sb=state_for(states,r['_A']),state_for(states,r['_B'])
  X.append(features(sa,sb)); y.append(int(r['_y']))
  update_elo(sa,sb,int(r['_y']))
  sa.update(int(r['_y']),fight_observation(r,'A'),r,'r'); sb.update(1-int(r['_y']),fight_observation(r,'B'),r,'b')
 return np.asarray(X),np.asarray(y),states

def _candidates():
 return {
  'logistic':Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.12,max_iter=5000))]),
  'extra_trees':ExtraTreesClassifier(n_estimators=700,min_samples_leaf=10,max_features=.65,class_weight='balanced',random_state=42,n_jobs=-1),
  'random_forest':RandomForestClassifier(n_estimators=700,min_samples_leaf=10,max_features=.65,class_weight='balanced',random_state=42,n_jobs=-1),
  'hist_gradient_boost':HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=15,learning_rate=.025,l2_regularization=2.0,random_state=42)
 }

def train(d):
 X,y,states=make_xy(d); n=len(X); cut=int(n*.65); val=int(n*.80)
 scores={}; models={}
 for name,m in _candidates().items():
  m.fit(X[:cut],y[:cut]); p=m.predict_proba(X[cut:val])[:,1]; scores[name]=float(log_loss(y[cut:val],np.clip(p,.001,.999))); models[name]=m
 best_name=min(scores,key=scores.get); best=_candidates()[best_name]; best.fit(X[:val],y[:val])
 raw=best.predict_proba(X[val:])[:,1]
 val_model=_candidates()[best_name]; val_model.fit(X[:cut],y[:cut]); vp=val_model.predict_proba(X[cut:val])[:,1]
 iso=IsotonicRegression(out_of_bounds='clip').fit(vp,y[cut:val]) if len(vp)>=50 else None
 cal=iso.predict(raw) if iso is not None else raw
 metrics={'n_total':len(y),'n_test':len(y[val:]),'accuracy':float(accuracy_score(y[val:],cal>=.5)),'raw_accuracy':float(accuracy_score(y[val:],raw>=.5)),'log_loss':float(log_loss(y[val:],np.clip(cal,.001,.999))),'brier':float(brier_score_loss(y[val:],cal)),'selected_model':best_name,'validation_logloss':scores[best_name],'candidate_logloss':scores}
 return best,iso,metrics,states

def save(bundle,path='data/model.joblib'): os.makedirs(os.path.dirname(path),exist_ok=True); joblib.dump(bundle,path)
def load(path='data/model.joblib'): return joblib.load(path)
def confidence(p,reliability=1.): return round(max(1,min(100,50+50*abs(p-.5)*2*reliability)),1)
def fair_american(p): return round(-100*p/(1-p)) if p>=.5 else round(100*(1-p)/p)
def implied_american(o): return -o/(-o+100) if o<0 else 100/(o+100)
