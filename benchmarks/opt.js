// Evaluate one configuration on the real page code. usage: node opt.js '<json>' <mem|res|both> <seed>
const {load,mulberry,ridgeMulti,pred,corr2,nrmse}=require('./core');
const cfg=JSON.parse(process.argv[2]),which=process.argv[3]||'both',seed=+(process.argv[4]||1);
const CON={rings:1,L:6,dz:0.3,gap:0.3,pillar:true,rc:0.3,gs:1,ori:'axial',arch:'C',multiSens:false,bands:1,longR:0,ctlR:1,calib:true,dotType:'d30',Bsens:2,...cfg.con};
const LNK={logicOn:true,linkSig:'change',linkMode:'exc',linkGain:1.0,jGain:0.5,linkDelay:3,...cfg.lnk};
const MEMD={gain:1,noise:0.05,D:0,curie:0,adapt:0,heat:0.3,linkGain:0,beta:3,...cfg.mem};
const RESD={gain:0.3,noise:0.01,D:1,curie:0,adapt:0,heat:0.3,inScale:1.2,...cfg.res};
function setup(){const d=load(seed);d.setMany(CON);d.setMany(LNK);d.get('build')();return d}
const out={cfg};const t0=Date.now();
if(which!=='res'){const d=setup();d.setMany({mode:'mem',...MEMD});const MP=cfg.memP||6;if(MP>6){const PATS=d.get('PATS');for(let i=7;i<=MP;i++)PATS.push(['r'+i,'R'+i]);const orig=d.get('pat');d.set('pat',(name,a)=>{if(/^r\d+$/.test(name)){const k=+name.slice(1);let h=(a+1)*2654435761^(k*40503+977);h=Math.imul(h^(h>>>15),2246822507);h^=h>>>13;return(h&1)?1:-1}return orig(name,a)})}
 const ts={layers:false,spiral:false,halves:false,rings:false,core:false};for(let i=1;i<=MP;i++)ts['r'+i]=true;d.set('toStore',ts);
 d.get('storePatterns')();let g=0;while(d.get('memJob')&&g<20000){d.get('progStep')();g++}
 const N=d.get('N'),EM=d.get('EM'),s=d.get('s'),ad=d.get('ad'),pat=d.get('pat'),r=mulberry(seed*17+3);const res=[];
 for(const dmg of [0.2,0.3])for(let m=1;m<=3;m++){for(let a=0;a<N;a++){if(!EM[a]){s[a]=0;continue}let v=pat('r'+m,a);if(r()<dmg)v=-v;s[a]=v*(0.6+0.4*r());ad[a]=0}d.set('ringLen',0);for(let k=0;k<150;k++)d.step();
  let o=0,n=0;for(let a=0;a<N;a++)if(EM[a]){o+=pat('r'+m,a)*Math.sign(s[a]);n++}res.push(o/n)}
 out.memP=MP;out.mem20=+(res.slice(0,3).reduce((x,y)=>x+y)/3).toFixed(3);out.mem30=+(res.slice(3).reduce((x,y)=>x+y)/3).toFixed(3);out.memScore=+((out.mem20+out.mem30)/2).toFixed(3);let tm=0;const Tg=d.get('Tg');for(let a=0;a<N;a++)tm=Math.max(tm,Tg[a]);out.memT=+tm.toFixed(2)}
if(which!=='mem'){const WASH=150,T=1400,tr=[],va=[],te=[];for(let t=WASH;t<WASH+T;t++){const q=t-WASH;(q<800?tr:q<1100?va:te).push(t)}const rnd=mulberry(seed*31+7);
 const mk=()=>{const d=setup();d.setMany({mode:'res',...RESD,linkGain:LNK.linkGain,jGain:LNK.jGain});return d};
 const pick=d=>{const N=d.get('N');let idx=[...Array(N).keys()];if(N>500){const r=mulberry(5);for(let i=N-1;i>0;i--){const j=Math.floor(r()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]]}idx=idx.slice(0,500)}return idx};
 const collect=(d,inp)=>{const idx=pick(d),X=[];let tm=0;for(let t=0;t<inp.length;t++){d.set('u1',inp[t][0]);d.set('u2',inp[t][1]);d.step();const s=d.get('s'),SS=d.get('SS'),EM=d.get('EM');const x=new Float64Array(idx.length+1);idx.forEach((a,i)=>x[i]=EM[a]?s[a]:SS[a]);x[idx.length]=1;X.push(x)}const Tg=d.get('Tg');for(let a=0;a<Tg.length;a++)tm=Math.max(tm,Tg[a]);out.resT=+tm.toFixed(2);return X};
 {const d=mk();const u=[];for(let t=0;t<WASH+T;t++)u.push([rnd()*2-1,0]);const X=collect(d,u);const Ks=[...Array(30).keys()].map(k=>k+1);const Ys=Ks.map(k=>u.map((v,t)=>t-k>=0?u[t-k][0]:NaN));const b=ridgeMulti(X,Ys,tr,va);
  out.MC=+Ks.map((k,i)=>corr2(te.map(t=>pred(b.Ws[i],X[t])),te.map(t=>Ys[i][t]))).reduce((a,c)=>a+c,0).toFixed(2)}
 {const tau=170,dt=0.1,NN=WASH+T+40;const xs=new Float32Array(NN*10+tau+35000);for(let t=0;t<=tau;t++)xs[t]=1.2;for(let t=tau;t<xs.length-1;t++){const xt=xs[t-tau];xs[t+1]=xs[t]+dt*(0.2*xt/(1+Math.pow(xt,10))-0.1*xs[t])}
  const off=20000+Math.floor(rnd()*1000)*10;const mg=[];for(let i=0;i<NN;i++)mg.push(xs[off+i*10]);const mn=Math.min(...mg),mx=Math.max(...mg);const m=mg.map(v=>2*(v-mn)/(mx-mn)-1);
  const d=mk();const X=collect(d,m.slice(0,WASH+T).map(v=>[v,0]));const Y=m.slice(0,WASH+T).map((v,t)=>m[t+5]);const b=ridgeMulti(X,[Y],tr,va);out.MG5=+nrmse(te.map(t=>pred(b.Ws[0],X[t])),te.map(t=>Y[t])).toFixed(3)}
 {const H=4,nb=Math.ceil((WASH+T)/H)+2,b1=[];for(let j=0;j<nb;j++)b1.push(rnd()<0.5?-1:1);const inp=[],blk=[];for(let t=0;t<WASH+T;t++){const j=Math.floor(t/H);inp.push([b1[j],0]);blk.push(j)}
  const d=mk();const X=collect(d,inp);const tx=blk.map(j=>j>0?(b1[j]*b1[j-1]<0?1:-1):NaN),p3=blk.map(j=>j>1?b1[j]*b1[j-1]*b1[j-2]:NaN);const b=ridgeMulti(X,[tx,p3],tr,va);const ends=te.filter(t=>t%H===H-1);
  const acc=(Wv,Y)=>ends.filter(t=>Math.sign(pred(Wv,X[t]))===Y[t]).length/ends.length;out.tXOR=+acc(b.Ws[0],tx).toFixed(3);out.par3=+acc(b.Ws[1],p3).toFixed(3)}
 out.compScore=+(0.4*Math.min(out.MC/8,1)+0.3*Math.max(0,1-out.MG5)+0.15*Math.max(0,(out.par3-0.5)*2)+0.15*Math.max(0,(out.tXOR-0.5)*2)).toFixed(3)}
out.sec=Math.round((Date.now()-t0)/1000);console.log(JSON.stringify(out));
