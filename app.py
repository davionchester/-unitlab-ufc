import os, datetime
import streamlit as st
import pandas as pd
from data import download_dataset, load, clean
from model import load as load_model, features, confidence, fair_american, implied_american

st.set_page_config(page_title='UnitLab UFC', page_icon='🥊', layout='wide')
st.title('🥊 UnitLab UFC')
st.caption('UFC-only • moneylines + props • Road to Victory • transparent model breakdown')
os.makedirs('data', exist_ok=True)

with st.sidebar:
    st.header('Model')
    if st.button('1. Build / refresh data', use_container_width=True):
        try:
            download_dataset(); st.success('Historical data downloaded.')
        except Exception as e: st.error(str(e))
    if st.button('2. Train + calibrate', use_container_width=True):
        try:
            from model import train, save
            d=clean(load()); pipe,iso,metrics,states=train(d)
            save({'pipe':pipe,'iso':iso,'metrics':metrics,'states':states}); st.success('Model trained.')
        except Exception as e: st.error(str(e))
    if os.path.exists('data/model.joblib'):
        m=load_model()['metrics']; st.success('Model ready')
        st.write(f"Fights: **{m['n_total']:,}**")
        st.write(f"Held-out accuracy: **{m['accuracy']*100:.1f}%**")
        st.write(f"Brier: **{m['brier']:.3f}**")
        if 'selected_model' in m: st.write(f"Selected model: **{m['selected_model']}**")

if not os.path.exists('data/model.joblib'):
    st.info('Start with the sidebar: **1. Build / refresh data**, then **2. Train + calibrate**.'); st.stop()

b=load_model(); states=b['states']; d=clean(load()); names=list(states)
latest_date=d['_date'].max() if d['_date'].notna().any() else None
st.caption(f"Training data through: **{latest_date.date() if pd.notna(latest_date) else 'unknown'}** • Prediction date: **{datetime.date.today()}**")
a,c=st.columns(2)
with a: fa_in=st.text_input('Fighter A',placeholder='Islam Makhachev')
with c: fb_in=st.text_input('Fighter B',placeholder='Charles Oliveira')

def resolve(q):
    q=q.lower().strip(); exact=[n for n in names if n.lower()==q]
    if exact:return exact[0]
    hits=[n for n in names if q in n.lower()]
    return hits[0] if hits else q

def find_historical(a,c):
    aa,cc=d['_A'].str.lower(),d['_B'].str.lower()
    return d[((aa==a.lower())&(cc==c.lower()))|((aa==c.lower())&(cc==a.lower()))].sort_values('_date')

def latest_profile(name):
    q=d[(d['_A'].str.lower()==name.lower())|(d['_B'].str.lower()==name.lower())]
    if q.empty:return {'wins':0,'losses':0,'win_streak':0,'sig':0,'sig_pct':0,'td':0,'td_pct':0,'sub_att':0,'height':0,'reach':0,'age':0,'rank':99,'shares':{'KO/TKO':0,'Submission':0,'Decision':1}}
    r=q.sort_values('_date').iloc[-1]; side='r' if str(r['_A']).lower()==name.lower() else 'b'
    def val(key,default=0.0):
        try:
            v=r.get(f'{side}_{key}',default); return float(v) if pd.notna(v) else default
        except:return default
    wins=max(0.0,val('wins')); ko=max(0.0,val('win_by_ko_tko')); sub=max(0.0,val('win_by_submission')); dec=max(0.0,wins-ko-sub); total=ko+sub+dec
    shares={'KO/TKO':ko/total,'Submission':sub/total,'Decision':dec/total} if total>0 else {'KO/TKO':0,'Submission':0,'Decision':1}
    return {'wins':wins,'losses':val('losses'),'win_streak':val('current_win_streak'),'sig':val('avg_sig_str_landed'),'sig_pct':val('avg_sig_str_pct'),'td':val('avg_td_landed'),'td_pct':val('avg_td_pct'),'sub_att':val('avg_sub_att'),'height':val('height_cms'),'reach':val('reach_cms'),'age':val('age'),'rank':val('match_weightclass_rank',99),'shares':shares}

def route(name,prob): return {k:float(prob*v) for k,v in latest_profile(name)['shares'].items()}

def factor_breakdown(sa,sb,fa,fb):
    A,B=sa.snapshot(),sb.snapshot()
    specs=[
      ('Elo / overall strength',A.get('elo',1500)-B.get('elo',1500),400,'Higher Elo reflects stronger historical performance.'),
      ('Historical win rate',A.get('winrate',.5)-B.get('winrate',.5),.20,'Better historical win rate supports the fighter.'),
      ('Recent 5-fight form',A.get('recent_win',.5)-B.get('recent_win',.5),.20,'Recent results are a form signal.'),
      ('Recent striking output',A.get('sig5',0)-B.get('sig5',0),5,'Higher recent striking production supports the fighter.'),
      ('Recent takedown output',A.get('td5',0)-B.get('td5',0),1,'Higher recent takedown production supports the fighter.'),
      ('Recent submission output',A.get('sub5',0)-B.get('sub5',0),.5,'Higher recent submission activity supports the fighter.'),
      ('Recent knockdown output',A.get('kd5',0)-B.get('kd5',0),.5,'Higher recent knockdown production supports the fighter.'),
      ('Experience',A.get('n',0)-B.get('n',0),3,'More recorded fights provide more historical evidence.'),
    ]
    rows=[]
    for label,diff,scale,why in specs:
        score=diff/scale
        if abs(score)<.20: edge='Neutral'; strength='Low'
        else: edge=fa if score>0 else fb; strength='High' if abs(score)>=.75 else 'Medium'
        rows.append({'Factor':label,'Edge':edge,'Strength':strength,'Why it matters':why})
    return rows

if fa_in and fb_in:
    fa,fb=resolve(fa_in),resolve(fb_in)
    if fa not in states or fb not in states: st.error('One or both fighters are not in the trained dataset.'); st.stop()
    historical=find_historical(fa,fb)
    if not historical.empty:
        r=historical.iloc[-1]
        st.warning('⚠️ **Historical matchup detected — this is NOT a live betting prediction.**')
        st.info(f"**Recorded result:** {r.get('_winner','Unknown')} • {r.get('method','')} • {r['_date'].date() if pd.notna(r['_date']) else 'unknown date'}")
    p=float(b['pipe'].predict_proba(features(states[fa],states[fb]).reshape(1,-1))[0,1])
    if b['iso'] is not None:p=float(b['iso'].predict([p])[0])
    p=max(.01,min(.99,p)); pb=1-p; winner=fa if p>=.5 else fb; wp=max(p,pb)
    rel=max(.55,min(1,1-b['metrics']['brier']*1.5)); conf=confidence(wp,rel)
    st.divider(); x,y,z=st.columns(3); x.metric(fa,f'{p*100:.1f}%'); y.metric('CONFIDENCE',f'{conf:.0f}%'); z.metric(fb,f'{pb*100:.1f}%')
    st.success(f'🏆 UNITLAB MODEL PICK: **{winner}** — {wp*100:.1f}%')

    st.subheader('🧠 Why the model picked this fighter')
    st.caption('This is a transparent, data-backed factor summary of the inputs used by the model. It is not a verbatim dump of hidden model reasoning.')
    factors=factor_breakdown(states[fa],states[fb],fa,fb)
    st.dataframe(pd.DataFrame(factors),use_container_width=True,hide_index=True)
    supporters=[r for r in factors if r['Edge']==winner and r['Strength']!='Low']
    counters=[r for r in factors if r['Edge'] not in (winner,'Neutral') and r['Strength']!='Low']
    st.markdown('**Top reasons for the pick**')
    if supporters:
        for r in supporters[:4]: st.write(f"• **{r['Factor']}** favors **{winner}** ({r['Strength']}). {r['Why it matters']}")
    else: st.write('• No individual factor has a strong enough edge to summarize as a primary driver.')
    if counters:
        st.markdown('**Counter-factors / what could beat the pick**')
        for r in counters[:3]: st.write(f"• **{r['Factor']}** favors **{r['Edge']}** ({r['Strength']}). This is a risk to the pick.")

    st.markdown('**Confidence breakdown**')
    st.write(f"• Model win probability: **{wp*100:.1f}%**")
    st.write(f"• Gap vs. opponent: **{abs(p-pb)*100:.1f} percentage points**")
    st.write(f"• Held-out calibration reliability factor: **{rel*100:.0f}%**")
    st.write(f"• Final UnitLab confidence: **{conf:.0f}/100**")

    st.subheader('💰 Moneyline value')
    if historical.empty:
        x,y=st.columns(2)
        for col,name,prob in [(x,fa,p),(y,fb,pb)]:
            with col:
                odds=st.number_input(f'{name} American odds',min_value=-1000,max_value=2000,value=0,step=5)
                if odds==0:st.caption('Enter sportsbook odds to calculate value.')
                else:
                    edge=prob-implied_american(odds); st.write(f'Fair odds: **{fair_american(prob)}**'); st.write(f'Implied: **{implied_american(odds)*100:.1f}%**'); st.write(f'Edge: **{edge*100:+.1f}%**'); st.success('VALUE' if edge>=.02 else 'PASS')
    else:st.info('Moneyline value is disabled for this historical matchup.')

    st.subheader('🛣️ Road to Victory')
    st.caption('Baseline route estimate = model win probability × historical finish mix. A dedicated finish-method classifier is a future accuracy upgrade.')
    ra,rb=route(fa,p),route(fb,pb); rows=[]
    for n,rp in [(fa,ra),(fb,rb)]:
        prof=latest_profile(n); rows.append({'Fighter':n,'KO/TKO':f"{rp['KO/TKO']*100:.1f}%",'Submission':f"{rp['Submission']*100:.1f}%",'Decision':f"{rp['Decision']*100:.1f}%",'Best route':max(rp,key=rp.get)})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    best=max([(v,n,k) for n,rp in [(fa,ra),(fb,rb)] for k,v in rp.items()]); st.info(f'⭐ **Highest-rated Road to Victory: {best[1]} by {best[2]} — {best[0]*100:.1f}%**')

    st.subheader('🎯 UFC props')
    prop=st.selectbox('Prop',[f'{fa} by KO/TKO',f'{fa} by Submission',f'{fa} by Decision',f'{fb} by KO/TKO',f'{fb} by Submission',f'{fb} by Decision','Fight goes the distance','Fight does NOT go the distance'])
    if prop.startswith(fa):prob=ra[prop.replace(f'{fa} by ','')]
    elif prop.startswith(fb):prob=rb[prop.replace(f'{fb} by ','')]
    else:
        prob=ra['Decision']+rb['Decision']; prob=prob if prop=='Fight goes the distance' else 1-prob
    st.metric('Model probability',f'{prob*100:.1f}%'); st.metric('Confidence',f'{confidence(prob,rel):.0f}%')
    st.caption('Prop probabilities are model-derived estimates and should be validated out-of-sample before being treated as betting edges.')

    st.subheader('📊 Fighter profile used')
    prof_rows=[]
    for n in [fa,fb]:
        q=latest_profile(n); prof_rows.append({'Fighter':n,'Wins':q['wins'],'Losses':q['losses'],'Win streak':q['win_streak'],'Sig/min':q['sig'],'Sig %':q['sig_pct'],'TD/15':q['td'],'TD %':q['td_pct'],'Sub attempts/15':q['sub_att'],'Height cm':q['height'],'Reach cm':q['reach'],'Age':q['age'],'Rank':q['rank']})
    st.dataframe(pd.DataFrame(prof_rows),use_container_width=True,hide_index=True)
