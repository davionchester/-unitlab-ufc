import os,streamlit as st,pandas as pd
from data import download_dataset,load,clean
from model import train,save,load as load_model,features,confidence,fair_american,implied_american
st.set_page_config(page_title='UnitLab UFC',page_icon='🥊',layout='wide')
st.title('🥊 UnitLab UFC'); st.caption('UFC-only • moneylines + props • Road to Victory')
os.makedirs('data',exist_ok=True)
with st.sidebar:
 st.header('Model')
 if st.button('1. Build / refresh data',use_container_width=True):
  try: download_dataset(); st.success('Historical data downloaded.')
  except Exception as e: st.error(str(e))
 if st.button('2. Train + calibrate',use_container_width=True):
  try:
   d=clean(load()); pipe,iso,metrics,states=train(d); save({'pipe':pipe,'iso':iso,'metrics':metrics,'states':states}); st.success('Model trained.')
  except Exception as e: st.error(str(e))
 if os.path.exists('data/model.joblib'):
  m=load_model()['metrics']; st.success('Model ready'); st.write(f"Fights: **{m['n_total']:,}**  "); st.write(f"Held-out accuracy: **{m['accuracy']*100:.1f}%**"); st.write(f"Brier: **{m['brier']:.3f}**")
if not os.path.exists('data/model.joblib'):
 st.info('Start with the sidebar: **1. Build / refresh data**, then **2. Train + calibrate**.'); st.stop()
b=load_model(); states=b['states']; names=list(states)
a,c=st.columns(2)
with a: fa=st.text_input('Fighter A',placeholder='Islam Makhachev')
with c: fb=st.text_input('Fighter B',placeholder='Charles Oliveira')
def resolve(q):
 q=q.lower().strip(); ex=[n for n in names if n.lower()==q];
 if ex:return ex[0]
 h=[n for n in names if q in n.lower()]; return h[0] if h else q
if fa and fb:
 fa,fb=resolve(fa),resolve(fb)
 if fa not in states or fb not in states: st.error('One or both fighters are not in the trained dataset.'); st.stop()
 p=float(b['pipe'].predict_proba(features(states[fa],states[fb]).reshape(1,-1))[0,1]);
 if b['iso'] is not None:p=float(b['iso'].predict([p])[0])
 p=max(.01,min(.99,p)); pb=1-p; winner=fa if p>=.5 else fb; wp=max(p,pb); rel=max(.55,min(1,1-b['metrics']['brier']*1.5)); conf=confidence(wp,rel)
 st.divider(); x,y,z=st.columns(3); x.metric(fa,f'{p*100:.1f}%'); y.metric('CONFIDENCE',f'{conf:.0f}%'); z.metric(fb,f'{pb*100:.1f}%'); st.success(f'🏆 UNITLAB PICK: **{winner}** — {wp*100:.1f}%')
 st.subheader('💰 Moneyline value'); x,y=st.columns(2)
 for col,name,prob in [(x,fa,p),(y,fb,pb)]:
  with col:
   odds=st.number_input(f'{name} American odds',-500,1000,-110 if prob>=.5 else 100,5); edge=prob-implied_american(odds); st.write(f'Fair odds: **{fair_american(prob)}**'); st.write(f'Implied: **{implied_american(odds)*100:.1f}%**'); st.write(f'Edge: **{edge*100:+.1f}%**'); st.success('VALUE' if edge>=.02 else 'PASS')
 st.subheader('🛣️ Road to Victory'); st.info('Current MVP uses each fighter’s historical finish profile. Dedicated method models are the next upgrade.')
 d=clean(load());
 def route(name,prob):
  q=d[(d['_A'].str.lower()==name.lower())|(d['_B'].str.lower()==name.lower())]; out={'KO/TKO':0,'Submission':0,'Decision':0}
  for _,r in q.iterrows():
   m=str(r.get('method','')).lower(); k='KO/TKO' if ('ko' in m or 'tko' in m) else 'Submission' if 'sub' in m else 'Decision' if 'dec' in m else None
   if k:out[k]+=1
  t=sum(out.values()) or 1; return {k:prob*v/t for k,v in out.items()}
 ra,rb=route(fa,p),route(fb,pb); rows=[]
 for n,r in [(fa,ra),(fb,rb)]: rows.append({'Fighter':n,**{k:f'{v*100:.1f}%' for k,v in r.items()},'Best route':max(r,key=r.get)})
 st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True); routes=[(v,n,k) for n,r in [(fa,ra),(fb,rb)] for k,v in r.items()]; best=max(routes); st.info(f'⭐ Highest-rated route: **{best[1]} by {best[2]} — {best[0]*100:.1f}%**')
 st.subheader('🎯 UFC props'); prop=st.selectbox('Prop',[f'{fa} by KO/TKO',f'{fa} by Submission',f'{fa} by Decision',f'{fb} by KO/TKO',f'{fb} by Submission',f'{fb} by Decision','Fight goes the distance','Fight does NOT go the distance'])
 if prop.startswith(fa): prob=ra[prop.replace(f'{fa} by ','')]
 elif prop.startswith(fb): prob=rb[prop.replace(f'{fb} by ','')]
 else: prob=ra['Decision']+rb['Decision']; prob=prob if prop=='Fight goes the distance' else 1-prob
 st.metric('Model probability',f'{prob*100:.1f}%'); st.metric('Confidence',f'{confidence(prob,rel):.0f}%')
