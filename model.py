import os, joblib, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score,log_loss,brier_score_loss
from data import fight_observation

class FighterState:
 def __init__(self): self.elo=1500.; self.n=0; self.wins=0; self.last=[]; self.stats={'sig':[],'td':[],'sub':[],'kd':[]}
 def snapshot(self):
  def avg(k): return float(np.mean(self.stats[k][-5:])) if self.stats[k] else 0.
  return {'elo':self.elo,'n':self.n,'winrate':self.wins/self.n if self.n else .5,'recent_win':np.mean(self.last[-5:]) if self.last else .5,'sig5':avg('sig'),'td5':avg('td'),'sub5':avg('sub'),'kd5':avg('kd')}
 def update(self,win,obs):
  self.n+=1; self.wins+=int(win); self.last.append(int(win))
  self.stats['sig'].append(obs.get('sig',0)); self.stats['td'].append(obs.get('td',0)); self.stats['sub'].append(obs.get('sub',0)); self.stats['kd'].append(obs.get('kd',0))
  self.last=self.last[-20:]
  for k in self.stats:self.stats[k]=self.stats[k][-20:]
def state_for(states,name):
 if name not in states: states[name]=FighterState()
 return states[name]
def features(sa,sb):
 a,b=sa.snapshot(),sb.snapshot()
 return np.array([(a['elo']-b['elo'])/400,a['winrate']-b['winrate'],a['recent_win']-b['recent_win'],(a['sig5']-b['sig5'])/50,(a['td5']-b['td5'])/5,(a['sub5']-b['sub5'])/2,(a['kd5']-b['kd5'])/2,min(a['n'],20)/20-min(b['n'],20)/20],float)
def update_elo(sa,sb,y):
 expected=1/(1+10**((sb.elo-sa.elo)/400)); k=28 if min(sa.n,sb.n)>5 else 36
 sa.elo+=k*(y-expected); sb.elo+=k*((1-y)-(1-expected))
def make_xy(d):
 states={}; X=[]; y=[]
 for _,r in d.iterrows():
  sa,sb=state_for(states,r['_A']),state_for(states,r['_B']); X.append(features(sa,sb)); y.append(int(r['_y']))
  update_elo(sa,sb,int(r['_y'])); sa.update(int(r['_y']),fight_observation(r,'A')); sb.update(1-int(r['_y']),fight_observation(r,'B'))
 return np.asarray(X),np.asarray(y),states

def _candidates():
 return {
  'logistic':Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.25,max_iter=5000))]),
  'extra_trees':ExtraTreesClassifier(n_estimators=500,min_samples_leaf=8,max_features=.8,class_weight='balanced',random_state=42,n_jobs=-1),
  'random_forest':RandomForestClassifier(n_estimators=500,min_samples_leaf=8,max_features=.8,class_weight='balanced',random_state=42,n_jobs=-1),
  'hist_gradient_boost':HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=15,learning_rate=.035,l2_regularization=1.5,random_state=42)
 }

def train(d):
 X,y,states=make_xy(d)
 n=len(X); cut=max(1,int(n*.70)); cal_cut=max(cut+1,int(n*.80));
 # Chronological model selection: choose the strongest model on a validation slice, then
 # refit that model on train+validation before evaluating once on the untouched final test.
 candidates=_candidates(); scores={}
 for name,model in candidates.items():
  model.fit(X[:cut],y[:cut]); pv=model.predict_proba(X[cut:cal_cut])[:,1]
  scores[name]=float(log_loss(y[cut:cal_cut],np.clip(pv,.001,.999)))
 best_name=min(scores,key=scores.get); best=_candidates()[best_name]
 best.fit(X[:cal_cut],y[:cal_cut]); raw=best.predict_proba(X[cal_cut:])[:,1]
 # Calibration is fitted only on the validation slice using the selected model trained on the early slice.
 val_model=_candidates()[best_name]; val_model.fit(X[:cut],y[:cut]); val_raw=val_model.predict_proba(X[cut:cal_cut])[:,1]
 iso=None
 if len(val_raw)>=50 and len(np.unique(y[cut:cal_cut]))>1: iso=IsotonicRegression(out_of_bounds='clip').fit(val_raw,y[cut:cal_cut]); cal=iso.predict(raw)
 else: cal=raw
 metrics={'n_total':len(y),'n_test':len(y[cal_cut:]),'accuracy':float(accuracy_score(y[cal_cut:],cal>=.5)),'raw_accuracy':float(accuracy_score(y[cal_cut:],raw>=.5)),'log_loss':float(log_loss(y[cal_cut:],np.clip(cal,.001,.999))),'brier':float(brier_score_loss(y[cal_cut:],cal)),'selected_model':best_name,'validation_logloss':scores[best_name],'candidate_logloss':scores}
 return best,iso,metrics,states

def save(bundle,path='data/model.joblib'): os.makedirs(os.path.dirname(path),exist_ok=True); joblib.dump(bundle,path)
def load(path='data/model.joblib'): return joblib.load(path)
def confidence(p,reliability=1.): return round(max(1,min(100,50+50*abs(p-.5)*2*reliability)),1)
def fair_american(p): return round(-100*p/(1-p)) if p>=.5 else round(100*(1-p)/p)
def implied_american(o): return -o/(-o+100) if o<0 else 100/(o+100)
