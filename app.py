import os, datetime
import streamlit as st
import pandas as pd
from data import download_dataset, load, clean
from model import train, save, load as load_model, features, confidence, fair_american, implied_american

st.set_page_config(page_title='UnitLab UFC', page_icon='🥊', layout='wide')
st.title('🥊 UnitLab UFC')
st.caption('UFC-only • moneylines + props • Road to Victory')
os.makedirs('data', exist_ok=True)

with st.sidebar:
    st.header('Model')
    if st.button('1. Build / refresh data', use_container_width=True):
        try:
            download_dataset()
            st.success('Historical data downloaded.')
        except Exception as e:
            st.error(str(e))
    if st.button('2. Train + calibrate', use_container_width=True):
        try:
            d = clean(load())
            pipe, iso, metrics, states = train(d)
            save({'pipe': pipe, 'iso': iso, 'metrics': metrics, 'states': states})
            st.success('Model trained.')
        except Exception as e:
            st.error(str(e))
    if os.path.exists('data/model.joblib'):
        m = load_model()['metrics']
        st.success('Model ready')
        st.write(f"Fights: **{m['n_total']:,}**")
        st.write(f"Held-out accuracy: **{m['accuracy']*100:.1f}%**")
        st.write(f"Brier: **{m['brier']:.3f}**")
        if 'selected_model' in m:
            st.write(f"Selected model: **{m['selected_model']}**")

if not os.path.exists('data/model.joblib'):
    st.info('Start with the sidebar: **1. Build / refresh data**, then **2. Train + calibrate**.')
    st.stop()

b = load_model()
states = b['states']
d = clean(load())
names = list(states)
latest_date = d['_date'].max() if d['_date'].notna().any() else None
st.caption(f"Training data through: **{latest_date.date() if pd.notna(latest_date) else 'unknown'}** • Prediction date: **{datetime.date.today()}**")

a, c = st.columns(2)
with a:
    fa_in = st.text_input('Fighter A', placeholder='Islam Makhachev')
with c:
    fb_in = st.text_input('Fighter B', placeholder='Charles Oliveira')

def resolve(q):
    q = q.lower().strip()
    exact = [n for n in names if n.lower() == q]
    if exact:
        return exact[0]
    hits = [n for n in names if q in n.lower()]
    return hits[0] if hits else q

def find_historical(a, c):
    aa, cc = d['_A'].str.lower(), d['_B'].str.lower()
    mask = ((aa == a.lower()) & (cc == c.lower())) | ((aa == c.lower()) & (cc == a.lower()))
    return d[mask].sort_values('_date')

def route(name, prob):
    q = d[(d['_A'].str.lower() == name.lower()) | (d['_B'].str.lower() == name.lower())]
    out = {'KO/TKO': 0, 'Submission': 0, 'Decision': 0}
    for _, r in q.iterrows():
        m = str(r.get('method', '')).lower()
        k = 'KO/TKO' if ('ko' in m or 'tko' in m) else 'Submission' if 'sub' in m else 'Decision' if 'dec' in m else None
        if k:
            out[k] += 1
    total = sum(out.values()) or 1
    return {k: prob * v / total for k, v in out.items()}

if fa_in and fb_in:
    fa, fb = resolve(fa_in), resolve(fb_in)
    if fa not in states or fb not in states:
        st.error('One or both fighters are not in the trained dataset.')
        st.stop()

    historical = find_historical(fa, fb)
    if not historical.empty:
        r = historical.iloc[-1]
        st.warning('⚠️ **Historical matchup detected — this is NOT a live betting prediction.** The matchup already exists in the training data. Use a future matchup to generate a live-style pick.')
        winner_recorded = str(r['_winner']) if str(r['_winner']) else 'Draw/No contest'
        method = str(r.get('method', ''))
        fight_date = r['_date'].date() if pd.notna(r['_date']) else 'unknown date'
        st.info(f"**Recorded result:** {winner_recorded} • {method} • {fight_date}")
        st.caption('This safeguard prevents the app from presenting a prediction for a fight that has already happened as if it were an upcoming wager.')

    p = float(b['pipe'].predict_proba(features(states[fa], states[fb]).reshape(1, -1))[0, 1])
    if b['iso'] is not None:
        p = float(b['iso'].predict([p])[0])
    p = max(.01, min(.99, p))
    pb = 1 - p
    winner = fa if p >= .5 else fb
    wp = max(p, pb)
    rel = max(.55, min(1, 1 - b['metrics']['brier'] * 1.5))
    conf = confidence(wp, rel)

    st.divider()
    x, y, z = st.columns(3)
    x.metric(fa, f'{p*100:.1f}%')
    y.metric('CONFIDENCE', f'{conf:.0f}%')
    z.metric(fb, f'{pb*100:.1f}%')
    st.success(f'🏆 UNITLAB MODEL PICK: **{winner}** — {wp*100:.1f}%')

    st.subheader('💰 Moneyline value')
    if historical.empty:
        x, y = st.columns(2)
        for col, name, prob in [(x, fa, p), (y, fb, pb)]:
            with col:
                odds = st.number_input(f'{name} American odds', min_value=-1000, max_value=2000, value=0, step=5)
                if odds == 0:
                    st.caption('Enter the sportsbook odds to calculate value.')
                else:
                    edge = prob - implied_american(odds)
                    st.write(f'Fair odds: **{fair_american(prob)}**')
                    st.write(f'Implied: **{implied_american(odds)*100:.1f}%**')
                    st.write(f'Edge: **{edge*100:+.1f}%**')
                    st.success('VALUE' if edge >= .02 else 'PASS')
    else:
        st.info('Moneyline value is disabled for this matchup because it is historical.')

    st.subheader('🛣️ Road to Victory')
    st.caption('Current version: historical finish-profile baseline. Dedicated method classifiers are the next model upgrade; these percentages should not be treated as independent market probabilities.')
    ra, rb = route(fa, p), route(fb, pb)
    rows = []
    for n, r in [(fa, ra), (fb, rb)]:
        rows.append({'Fighter': n, **{k: f'{v*100:.1f}%' for k, v in r.items()}, 'Best route': max(r, key=r.get)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    routes = [(v, n, k) for n, r in [(fa, ra), (fb, rb)] for k, v in r.items()]
    best = max(routes)
    st.info(f'⭐ Highest-rated route: **{best[1]} by {best[2]} — {best[0]*100:.1f}%**')

    st.subheader('🎯 UFC props')
    prop = st.selectbox('Prop', [f'{fa} by KO/TKO', f'{fa} by Submission', f'{fa} by Decision', f'{fb} by KO/TKO', f'{fb} by Submission', f'{fb} by Decision', 'Fight goes the distance', 'Fight does NOT go the distance'])
    if prop.startswith(fa):
        prob = ra[prop.replace(f'{fa} by ', '')]
    elif prop.startswith(fb):
        prob = rb[prop.replace(f'{fb} by ', '')]
    else:
        prob = ra['Decision'] + rb['Decision']
        prob = prob if prop == 'Fight goes the distance' else 1 - prob
    st.metric('Model probability', f'{prob*100:.1f}%')
    st.metric('Confidence', f'{confidence(prob, rel):.0f}%')

    if historical.empty:
        st.success('LIVE MODE: matchup not found in historical data.')
    else:
        st.error('BACKTEST / HISTORICAL MODE: do not use this result as a current betting signal.')
