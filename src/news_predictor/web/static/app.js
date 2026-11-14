'use strict';
const $ = (id) => document.getElementById(id);
const fmt = (num, dp=2) => Number.isFinite(Number(num)) ? Number(num).toLocaleString('en-US', {minimumFractionDigits: dp, maximumFractionDigits: dp}) : 'Unavailable';
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const directionLabel = (s) => s === 'up' ? 'BULLISH ↗' : s === 'down' ? 'BEARISH ↘' : s === 'flat' ? 'NEUTRAL' : 'UNAVAILABLE';
const appState = { mode:'demo', sort:'mentions', selected:'0700.HK', overview:null, detail:null };
async function api(path) {const response = await fetch(path, {headers:{'Accept':'application/json'}}); const body = await response.json(); if(!response.ok) throw Error(body.detail || `HTTP ${response.status}`); return body;}
function showError(message) { $('noticeTitle').textContent = 'DATA UNAVAILABLE'; $('noticeText').textContent = ` ${message}`; }
function setNotice(data) { $('noticeTitle').textContent = data.is_demo ? 'SYNTHETIC DEMONSTRATION' : 'LOCAL RESEARCH MODE'; $('noticeText').textContent = ` ${data.notice}`; $('asOf').textContent = data.as_of ? `As of ${data.as_of}` : 'No local data available'; $('upFoot').textContent = data.is_demo ? 'Synthetic directions only' : 'Research results only'; }
function setStats(data) { $('statAssets').textContent = `${data.assets_available} / ${data.assets_configured}`; $('statMentions').textContent = data.total_social_mentions.toLocaleString('en-US'); $('statPositive').textContent = String(data.positive_sentiment_assets); $('statUp').textContent = String(data.upward_signals); }
function setTable(items) {
  $('assetRows').innerHTML = items.length ? items.map(item => {
    let sentiment = item.social_sentiment === null ? '—' : (item.social_sentiment >= 0 ? '+' : '') + fmt(item.social_sentiment, 2);
    let signClass = item.social_sentiment === null ? '' : item.social_sentiment >= 0 ? 'sentiment-up':'sentiment-down';
    return `<tr class="clickable ${appState.selected === item.ticker ? 'selected' : ''}" data-ticker="${esc(item.ticker)}" tabindex="0" role="button" aria-label="Inspect ${esc(item.asset_name)}"><td><span class="table-primary">${esc(item.asset_name)}</span><span class="table-secondary">${esc(item.ticker)}</span></td><td><span class="val-num">${fmt(item.social_mentions,0)}</span></td><td><span class="val-num ${signClass}">${sentiment}</span></td><td><span class="direction ${esc(item.direction)}">${directionLabel(item.direction)}</span></td><td><span class="open-asset">›</span></td></tr>`;
  }).join('') : `<tr><td colspan="5" class="empty-row">No saved real-data results found. Run the notebook and prepare local CSVs, or switch to Synthetic demo.</td></tr>`;
  document.querySelectorAll('[data-ticker]').forEach(row=>{
    row.addEventListener('click', ()=> selectTicker(row.dataset.ticker));
    row.addEventListener('keydown', e => {if(e.key === 'Enter' || e.key === ' ') {e.preventDefault(); selectTicker(row.dataset.ticker);}});
  });
}
function setOptions(items) {const current=appState.selected; $('tickerSelect').innerHTML = items.map(x=>`<option value="${esc(x.ticker)}">${esc(x.ticker)} · ${esc(x.asset_name)}</option>`).join(''); if(items.some(a=>a.ticker===current)) $('tickerSelect').value=current; else if(items.length) { appState.selected=items[0].ticker; $('tickerSelect').value=appState.selected; } }
function svgLine(rows, lines, height=255, yBounds=null) {
  const W=760,H=height,L=46,R=18,T=15,B=30; const n=rows.length;
  if(!n) return '<div class="loading-text">No time series available</div>';
  const values=lines.flatMap(x=>rows.map(r=>r[x.key])).filter(v=>Number.isFinite(Number(v))).map(Number);
  if(!values.length) return '<div class="loading-text">No numeric observations</div>';
  let lo=yBounds ? yBounds[0] : Math.min(...values), hi=yBounds ? yBounds[1] : Math.max(...values);
  const pad=Math.max((hi-lo)*.1,hi === lo ? 1 : .01);lo-=pad;hi+=pad;
  const x=(i)=> L+(W-L-R)*(n===1?.5:i/(n-1));const y=(v)=>T+(H-T-B)*(hi-Number(v))/(hi-lo);
  const path=(key)=> rows.map((r,i)=>Number.isFinite(Number(r[key]))?`${i===0?'M':'L'}${x(i).toFixed(2)} ${y(r[key]).toFixed(2)}`:'').filter(Boolean).join(' ');
  const grids=[0,.25,.5,.75,1].map(p=>{let yy=T+p*(H-T-B);let price=hi-p*(hi-lo);return `<line class="grid-line" x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}"/><text class="chart-axis" x="2" y="${yy+4}">${fmt(price,1)}</text>`;}).join('');
  const dateLabel=(idx)=>esc(String(rows[idx].session || '').slice(5));
  const axis=[0,Math.floor((n-1)/2),n-1].map(i=>`<text class="chart-axis" x="${x(i)}" y="${H-7}" text-anchor="middle">${dateLabel(i)}</text>`).join('');
  const curves=lines.map(line=>`<path class="chart-line ${line.color}" d="${path(line.key)}"/>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Line chart of ${lines.map(x=>x.key).join(', ')}"><g>${grids}${axis}${curves}</g></svg>`;
}
function svgBars(rows) {
  const n=Math.min(rows.length,30),data=rows.slice(-n); if(!n)return 'No available mentions';
  const W=600,H=130;let max=Math.max(...data.map(r=>r.mentions),1);const spacing=(W-20)/n;
  let bars=data.map((r,i)=>{let h=(Math.max(0,r.mentions)/max)*97;return `<rect x="${10+i*spacing+1}" y="${110-h}" width="${Math.max(2,spacing-3)}" height="${h}" rx="2" fill="${i===n-1?'#42d6b5':'#2b776e'}"><title>${esc(r.session)} · ${r.mentions} mentions</title></rect>`;}).join('');
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Recent daily social mentions"><line x1="10" y1="110" x2="590" y2="110" stroke="#3b4a5c"/>${bars}<text x="10" y="128" class="chart-axis">${esc(data[0].session)}</text><text x="590" y="128" text-anchor="end" class="chart-axis">${esc(data[n-1].session)}</text></svg>`;
}
function updateDetail(d) {
  appState.detail=d;
  $('assetName').textContent=d.asset_name; $('assetSub').textContent=`${d.exchange} · ${d.ticker} · ${d.is_demo?'SYNTHETIC':'RESEARCH DATA'}`;
  $('lastPrice').textContent=d.current_close != null ? `HK$ ${fmt(d.current_close)}`:'Unavailable';
  const history=d.history||[];const previous=history.length>1?history[history.length-2].close:null;
  const change=previous&&d.current_close ? 100*(d.current_close/previous-1):null;
  $('priceChange').textContent=change == null ? 'No prior close' : `${change>=0?'+':''}${fmt(change)}% vs previous session`;
  const range=history.filter(r=>Number.isFinite(r.close)).map(r=>({...r, support:d.support, resistance:d.resistance}));$('priceChart').innerHTML=svgLine(range,[{key:'close',color:'teal'},{key:'support',color:'gold'},{key:'resistance',color:'lilac'}]);$('chartRange').textContent=`${history.length} observed sessions`;
  const hasForecast=d.predicted_next_close != null;
  $('forecastStatus').textContent=d.is_demo?'Synthetic estimate':hasForecast?'Research forecast':'No verified forecast';
  $('forecastPrice').textContent=hasForecast?`HK$ ${fmt(d.predicted_next_close)}`:'Unavailable';
  $('directionBadge').textContent=directionLabel(d.direction);$('directionBadge').className=`direction-tag ${d.direction}`;
  $('forecastReturn').textContent=d.predicted_return==null?'—':`${d.predicted_return>=0?'+':''}${fmt(d.predicted_return*100)}% forecast return`;
  $('support').textContent=d.support==null?'Unavailable':`HK$ ${fmt(d.support)}`;
  $('resistance').textContent=d.resistance==null?'Unavailable':`HK$ ${fmt(d.resistance)}`;
  $('position').textContent=d.position?.replaceAll('_',' ')||'None';$('sessionDate').textContent=d.as_of_session||'Unavailable';
  $('forecastExplain').textContent=d.is_demo?'SYNTHETIC — generated values for interface testing only. No trained model and no investment signal.': d.prediction_status;
  const sources=d.source_mentions||{};
  const breakdown=[{name:'LIHKG',value:sources.lihkg||0,style:'primary'}, {name:'Futu',value:sources.futu||0,style:'secondary'}, {name:'Exa news',value:d.news_mentions||0,style:'news'}];
  let max=Math.max(1,...breakdown.map(s=>s.value));
  $('sentimentBreakdown').innerHTML=breakdown.map(src=>`<div class="signal-row"><span>${esc(src.name)}</span><strong>${fmt(src.value,0)} ${src.name==='Exa news'?'articles':'posts'}</strong></div><div class="bar-track"><div class="bar-fill ${src.style==='primary'?'':src.style}" style="width:${Math.round(100*src.value/max)}%"></div></div>`).join('')+`<p class="sentiment-total">Social score: ${d.social_sentiment==null?'Unavailable':fmt(d.social_sentiment,3)} · News score: ${d.news_sentiment==null?'Unavailable':fmt(d.news_sentiment,3)}. Scale: −1 to +1.</p>`;
  $('mentionsChart').innerHTML=svgBars(history);
  const evaluation=d.evaluation;
  if(evaluation && evaluation.test_series && evaluation.test_series.length){
    $('evaluationPill').textContent=d.is_demo?'Synthetic backtest illustration':'Saved out-of-sample evaluation';
    $('evalDescription').textContent=d.is_demo?'Example test series generated from synthetic data — not a financial performance result.':'Saved held-out predictions from the local research notebook.';
    $('evaluationChart').innerHTML=svgLine(evaluation.test_series,[{key:'actual',color:'teal'},{key:'predicted',color:'lilac'}],234);
    $('metricMAE').textContent=`HK$ ${fmt(evaluation.price_mae)}`;
    $('metricRMSE').textContent=`HK$ ${fmt(evaluation.price_rmse)}`;
    $('metricBaseline').textContent=`HK$ ${fmt(evaluation.baseline_mae)}`;
    $('metricObservations').textContent=String(evaluation.sample_count??'—');
    $('metricsDisclaimer').textContent=d.is_demo?'SYNTHETIC METRICS — illustration only, not a real trained model or proof of profitability.':'LOCAL HELD-OUT RESEARCH METRICS — check sample size, timestamps, leakage and data permissions before drawing conclusions.';
  } else {
    $('evaluationPill').textContent='Not evaluated'; $('evalDescription').textContent='No saved evaluation.json is available for this ticker.';
    $('evaluationChart').innerHTML='<div class="loading-text">Run and validate training in the notebook to generate real evaluation artifacts.</div>';
    for(const id of ['metricMAE','metricRMSE','metricBaseline','metricObservations'])$(id).textContent='—';
    $('metricsDisclaimer').textContent='No metrics are invented when real training results are absent.';
  }
}
async function selectTicker(ticker){appState.selected=ticker; $('tickerSelect').value=ticker; if(appState.overview) setTable(appState.overview.assets); try{const detail=await api(`/api/assets/${encodeURIComponent(ticker)}/analysis?mode=${appState.mode}`); if(ticker===appState.selected)updateDetail(detail);}catch(e){showError(e.message);emptyDetail();}}
function emptyDetail(){['assetName','assetSub','lastPrice','priceChange','forecastPrice','support','resistance','position','sessionDate','forecastReturn','metricMAE','metricRMSE','metricBaseline','metricObservations'].forEach(id=>$(id).textContent='—');$('priceChart').innerHTML='<span>No local dataset found for this asset.</span>';$('evaluationChart').innerHTML='<span>No evaluation saved.</span>';$('sentimentBreakdown').innerHTML='<span class="loading-text">Unavailable</span>';$('mentionsChart').innerHTML='';$('directionBadge').textContent='UNAVAILABLE';$('forecastExplain').textContent='Switch to synthetic demo or run the notebook to generate real local data.';}
async function loadOverview(){try{const data=await api(`/api/overview?mode=${appState.mode}&sort_by=${appState.sort}`);appState.overview=data;setNotice(data);setStats(data);setTable(data.assets);setOptions(data.assets);if(data.assets.length){await selectTicker(appState.selected);}else emptyDetail();}catch(e){showError(e.message);emptyDetail();}}
async function loadSources(){try{let d=await api('/api/sources');$('sources').innerHTML=d.sources.map((src,i)=>`<div class="source-row"><div class="source-circle">${['L','F','E','Y'][i]||'•'}</div><div class="source-main"><strong>${esc(src.name)}</strong><small>${esc(src.integration)} · ${esc(src.status.replaceAll('_',' '))}</small></div><div class="source-dot" title="Access/rights validation required"></div></div>`).join('');}catch(e){$('sources').textContent='Cannot load source details.';}}
$('modeSelect').addEventListener('change',ev=>{appState.mode=ev.target.value;loadOverview();});$('sortSelect').addEventListener('change',ev=>{appState.sort=ev.target.value;loadOverview();});$('tickerSelect').addEventListener('change',ev=>{appState.selected=ev.target.value;selectTicker(appState.selected);});
$('clock').textContent=new Date().toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'}).toUpperCase();loadSources();loadOverview();
