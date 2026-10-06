// Strong echo-state-network baseline on identical tasks. usage: node esn2.js '<json {rho,leak,ins,nin}>' <seed>
const {mulberry,ridgeMulti,pred,corr2,nrmse}=require('./core');
const cfg=JSON.parse(process.argv[2]),seed=+(process.argv[3]||1),which='res';const out={cfg};const t0=Date.now();const NE=cfg.N||756;
function mkESN(){const r=mulberry(seed*977+3);const W=[];for(let i=0;i<NE;i++){const row=[];for(let j=0;j<NE;j++)if(r()<0.1)row.push([j,r()*2-1]);W.push(row)}
 let v=new Float64Array(NE).fill(1),lam=1;for(let it=0;it<60;it++){const nv=new Float64Array(NE);for(let i=0;i<NE;i++){let a=0;for(const [j,w] of W[i])a+=w*v[j];nv[i]=a}let nn=Math.sqrt(nv.reduce((a,b)=>a+b*b,0)),on=Math.sqrt(v.reduce((a,b)=>a+b*b,0));lam=nn/on;for(let i=0;i<NE;i++)v[i]=nv[i]/nn}
 const sc=cfg.rho/lam;W.forEach(row=>row.forEach(e=>e[1]*=sc));const nin=cfg.nin||NE;const Win=new Float64Array(NE),bias=new Float64Array(NE);for(let i=0;i<NE;i++){Win[i]=i<nin?(r()*2-1)*cfg.ins:0;bias[i]=(r()*2-1)*0.5}
 const x=new Float64Array(NE),nr=mulberry(seed*13+1);
 return{get:k=>k==='N'?NE:null,set:(k,v)=>{if(k==='u1')this_u=v},step(){const nx=new Float64Array(NE);for(let i=0;i<NE;i++){let a=Win[i]*this_u+bias[i];for(const [j,w] of W[i])a+=w*x[j];a+=0.01*(nr()*2-1);nx[i]=(1-cfg.leak)*x[i]+cfg.leak*Math.tanh(a)}x.set(nx)},x}}
let this_u=0;
if(which!=='mem'){const WASH=150,T=1400,tr=[],va=[],te=[];for(let t=WASH;t<WASH+T;t++){const q=t-WASH;(q<800?tr:q<1100?va:te).push(t)}const rnd=mulberry(seed*31+7);
 const mk=()=>mkESN();
 const pick=d=>{const N=NE;let idx=[...Array(N).keys()];if(N>500){const r=mulberry(5);for(let i=N-1;i>0;i--){const j=Math.floor(r()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]]}idx=idx.slice(0,500)}return idx};
 const collect=(d,inp)=>{const idx=pick(d),X=[];for(let t=0;t<inp.length;t++){this_u=inp[t][0];d.step();const x=new Float64Array(idx.length+1);idx.forEach((a,i)=>x[i]=d.x[a]);x[idx.length]=1;X.push(x)}return X};
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
