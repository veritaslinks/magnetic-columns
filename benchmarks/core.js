// Headless harness: loads the real page script, exposes internals, seeded randomness.
const {JSDOM}=require('jsdom');const fs=require('fs');const R3=require('three');const path=require('path');
const HTML=fs.readFileSync(path.join(__dirname,'..','index.html'),'utf8').replace(/<script src=[^>]*><\/script>/,'');
const SCR=HTML.match(/<script>([\s\S]*?)<\/script>/g).pop().replace(/<\/?script>/g,'');
function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function load(seed,opts={}){const PATCH=opts.patch||(x=>x);
 const dom=new JSDOM(HTML,{runScripts:'outside-only',pretendToBeVisual:true});const w=dom.window;
 w.Math.random=mulberry(seed*7919+13);
 w.THREE=new Proxy(R3,{get(t,k){if(k==='WebGLRenderer')return function(){const c=w.document.createElement('canvas');return {domElement:c,setPixelRatio(){},setSize(){},render(){}}};return t[k]}});
 w.ResizeObserver=class{observe(){}};w.matchMedia=()=>({matches:false,addEventListener(){}});
 w.HTMLCanvasElement.prototype.getContext=()=>new Proxy({},{get:(t,k)=>()=>{},set:()=>true});
 w.requestAnimationFrame=()=>{};w.setTimeout=()=>{};
 w.__Q=x=>x;
 let scr=PATCH(SCR.replace('effW[a]=s[a]*w;','effW[a]=window.__Q(s[a]*w);'));
 if(scr===SCR)throw new Error('quantization hook not applied');
 w.eval(scr.replace('\nframe();\n','\nwindow.__d={step,build,computeJ,get:(k)=>eval(k),set:(k,v)=>{window.__v=v;eval(k+"=window.__v")}};\n'));
 const d=w.__d;d.w=w;
 d.setMany=o=>{for(const k in o)d.set(k,o[k])};
 return d;
}
// ---------- linear algebra ----------
function chol(A,n){const L=new Float64Array(n*n);for(let i=0;i<n;i++){for(let j=0;j<=i;j++){let s=A[i*n+j];for(let k=0;k<j;k++)s-=L[i*n+k]*L[j*n+k];if(i===j)L[i*n+i]=Math.sqrt(Math.max(s,1e-12));else L[i*n+j]=s/L[j*n+j]}}return L}
function cholSolve(L,b,n){const y=new Float64Array(n);for(let i=0;i<n;i++){let s=b[i];for(let k=0;k<i;k++)s-=L[i*n+k]*y[k];y[i]=s/L[i*n+i]}const x=new Float64Array(n);for(let i=n-1;i>=0;i--){let s=y[i];for(let k=i+1;k<n;k++)s-=L[k*n+i]*x[k];x[i]=s/L[i*n+i]}return x}
// X: rows of features (Float64Array each, last = bias). returns predictions for all rows for several targets
function ridgeMulti(X,Ys,tr,va,lams=[1e-3,1e-2,1e-1,1,10]){const n=X[0].length;const G=new Float64Array(n*n);
 for(const t of tr){const x=X[t];for(let i=0;i<n;i++){const xi=x[i];if(xi===0)continue;for(let j=0;j<=i;j++)G[i*n+j]+=xi*x[j]}}
 for(let i=0;i<n;i++)for(let j=0;j<i;j++)G[j*n+i]=G[i*n+j];
 const XtY=Ys.map(Y=>{const v=new Float64Array(n);for(const t of tr){const x=X[t],y=Y[t];if(isNaN(y))continue;for(let i=0;i<n;i++)v[i]+=x[i]*y}return v});
 let best=null;for(const lam of lams){const A=Float64Array.from(G);for(let i=0;i<n-1;i++)A[i*n+i]+=lam*tr.length/100;const L=chol(A,n);
  const Ws=XtY.map(v=>cholSolve(L,v,n));let err=0;Ws.forEach((wv,k)=>{for(const t of va){const y=Ys[k][t];if(isNaN(y))continue;let p=0;const x=X[t];for(let i=0;i<n;i++)p+=wv[i]*x[i];err+=(p-y)**2}});
  if(!best||err<best.err)best={err,Ws,lam}}
 return best}
const pred=(wv,x)=>{let p=0;for(let i=0;i<x.length;i++)p+=wv[i]*x[i];return p};
function corr2(a,b){const n=a.length;let ma=0,mb=0;for(let i=0;i<n;i++){ma+=a[i];mb+=b[i]}ma/=n;mb/=n;let sa=0,sb=0,sab=0;for(let i=0;i<n;i++){sa+=(a[i]-ma)**2;sb+=(b[i]-mb)**2;sab+=(a[i]-ma)*(b[i]-mb)}return sa*sb>0?sab*sab/(sa*sb):0}
function nrmse(p,y){let m=0;y.forEach(v=>m+=v);m/=y.length;let v=0,e=0;for(let i=0;i<y.length;i++){v+=(y[i]-m)**2;e+=(p[i]-y[i])**2}return Math.sqrt(e/Math.max(v,1e-12))}
module.exports={load,mulberry,ridgeMulti,pred,corr2,nrmse};
