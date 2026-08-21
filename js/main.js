(function(){

  function mulberry32(seed){
    let t = seed >>> 0;
    return function(){
      t += 0x6D2B79F5;
      let r = Math.imul(t ^ t>>>15, 1 | t);
      r ^= r + Math.imul(r ^ r>>>7, 61 | r);
      return ((r ^ r>>>14) >>> 0) / 4294967296;
    };
  }

  function applyActivation(s, activation, p){
    switch(activation){
      case 'linear': {
        const slope = p.slope;
        for (let i=0;i<s.length;i++) s[i] = slope * s[i];
        return s;
      }
      case 'sin': {
        const freq = p.freq;
        const phase = p.phase;
        for (let i=0;i<s.length;i++) s[i] = Math.sin(freq * s[i] + phase);
        return s;
      }
      case 'sin2': {
        const freq = p.sin2Freq;
        const phase = p.sin2Phase;
        for (let i=0;i<s.length;i++){
          const v = Math.sin(freq * s[i] + phase);
          s[i] = v * v;
        }
        return s;
      }
      case 'sin3': {
        const freq = p.sin3Freq;
        const phase = p.sin3Phase;
        for (let i=0;i<s.length;i++){
          const v = Math.sin(freq * s[i] + phase);
          s[i] = v * v * v;
        }
        return s;
      }
      case 'power': {
        const pp = p.p;
        for (let i=0;i<s.length;i++){
          const v = s[i];
          s[i] = Math.sign(v) * Math.pow(Math.abs(v), pp);
        }
        return s;
      }
      case 'tanh': {
        const mult = p.tanhMult;
        for (let i=0;i<s.length;i++) s[i] = Math.tanh(mult * s[i]);
        return s;
      }
      case 'abs': {
        const absSlope = p.absSlope;
        for (let i=0;i<s.length;i++) s[i] = Math.abs(absSlope * s[i]);
        return s;
      }
      case 'x2': {
        const coef = p.x2Coef;
        for (let i=0;i<s.length;i++) s[i] = coef * s[i] * s[i];
        return s;
      }
      case 'x3': {
        const coef = p.x3Coef;
        for (let i=0;i<s.length;i++) s[i] = coef * s[i] * s[i] * s[i];
        return s;
      }
      case 'inv_gaussian': {
        const mu = p.mu, sigma = p.sigma;
        for (let i=0;i<s.length;i++){
          const v = s[i];
          s[i] = 1.0 - Math.exp(-((v-mu)*(v-mu)) / (2*sigma*sigma));
        }
        return s;
      }
      default: throw new Error('Unknown activation: ' + activation);
    }
  }

  function generateContinuousCA(cfg){
    const {N, M, seed, wrap, mode, activation, activationParams, mask, init, impulsePos, nCells} = cfg;

    const X = new Float64Array(M * N); // typed array is zero-filled by default

    if (init === 'impulse'){
      const frac = Number.isFinite(impulsePos) ? impulsePos : 0.5;
      const idx = Math.min(N - 1, Math.max(0, Math.floor(frac * N)));
      X[idx] = 1.0;
    } else if (init === 'random_n'){
      const rng = (seed === null || seed === undefined || isNaN(seed)) ? Math.random : mulberry32(seed);
      const n = Math.min(N, Math.max(1, Math.round(Number.isFinite(nCells) ? nCells : 1)));
      // partial Fisher-Yates shuffle to pick n distinct indices in [0,N)
      const idxPool = new Int32Array(N);
      for (let j=0;j<N;j++) idxPool[j] = j;
      for (let i=0;i<n;i++){
        const j = i + Math.floor(rng() * (N - i));
        const tmp = idxPool[i]; idxPool[i] = idxPool[j]; idxPool[j] = tmp;
      }
      for (let i=0;i<n;i++) X[idxPool[i]] = rng();
    } else {
      const rng = (seed === null || seed === undefined || isNaN(seed)) ? Math.random : mulberry32(seed);
      for (let j=0;j<N;j++) X[j] = rng();
    }

    for (let i=1;i<M;i++){
      const prevOff = (i-1)*N;
      const rowOff = i*N;
      const s = new Float64Array(N);

      for (let j=0;j<N;j++){
        let leftVal, rightVal;
        if (wrap){
          leftVal  = X[prevOff + ((j-1+N) % N)];
          rightVal = X[prevOff + ((j+1) % N)];
        } else {
          leftVal  = (j-1 >= 0) ? X[prevOff + j - 1] : 0;
          rightVal = (j+1 < N)  ? X[prevOff + j + 1] : 0;
        }
        const centerVal = X[prevOff + j];
        s[j] = mask[0]*leftVal + mask[1]*centerVal + mask[2]*rightVal;
      }

      const a = applyActivation(s, activation, activationParams);

      for (let j=0;j<N;j++){
        let v = a[j];
        if (mode === 'mod'){
          v = v - Math.floor(v);
        } else {
          v = v < 0 ? 0 : (v > 1 ? 1 : v);
        }
        X[rowOff + j] = v;
      }
    }
    return X;
  }

  function nextPow2(x){ let p = 1; while (p < x) p <<= 1; return p; }

  function fftRadix2(re, im, invert){
    const n = re.length;
    for (let i=1, j=0; i<n; i++){
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j){
        let t = re[i]; re[i]=re[j]; re[j]=t;
        t = im[i]; im[i]=im[j]; im[j]=t;
      }
    }
    for (let len=2; len<=n; len<<=1){
      const half = len>>1;
      const ang = (invert ? 2 : -2) * Math.PI / len;
      const wlenRe = Math.cos(ang), wlenIm = Math.sin(ang);
      for (let i=0;i<n;i+=len){
        let wRe=1, wIm=0;
        for (let j=0;j<half;j++){
          const uRe = re[i+j], uIm = im[i+j];
          const tRe = re[i+j+half]*wRe - im[i+j+half]*wIm;
          const tIm = re[i+j+half]*wIm + im[i+j+half]*wRe;
          re[i+j] = uRe+tRe; im[i+j]=uIm+tIm;
          re[i+j+half] = uRe-tRe; im[i+j+half]=uIm-tIm;
          const nwRe = wRe*wlenRe - wIm*wlenIm;
          const nwIm = wRe*wlenIm + wIm*wlenRe;
          wRe = nwRe; wIm = nwIm;
        }
      }
    }
    if (invert){ for (let i=0;i<n;i++){ re[i]/=n; im[i]/=n; } }
  }

  function makeBluesteinPlan(n){
    const m = nextPow2(2*n - 1);
    const chirpRe = new Float64Array(n);
    const chirpIm = new Float64Array(n);
    const twoN = 2*n;
    for (let k=0;k<n;k++){
      const kk = (k*k) % twoN;
      const ang = -Math.PI*kk/n;
      chirpRe[k] = Math.cos(ang);
      chirpIm[k] = Math.sin(ang);
    }
    const BwRe = new Float64Array(m);
    const BwIm = new Float64Array(m);
    for (let k=0;k<n;k++){ BwRe[k]=chirpRe[k]; BwIm[k]=-chirpIm[k]; }
    for (let k=1;k<n;k++){ BwRe[m-k]=chirpRe[k]; BwIm[m-k]=-chirpIm[k]; }
    fftRadix2(BwRe, BwIm, false);
    return {n, m, chirpRe, chirpIm, BwRe, BwIm, aRe:new Float64Array(m), aIm:new Float64Array(m)};
  }

  function bluesteinForwardReal(plan, xRe, outRe, outIm){
    const {n, m, chirpRe, chirpIm, BwRe, BwIm, aRe, aIm} = plan;
    aRe.fill(0); aIm.fill(0);
    for (let j=0;j<n;j++){
      aRe[j] = xRe[j]*chirpRe[j];
      aIm[j] = xRe[j]*chirpIm[j];
    }
    fftRadix2(aRe, aIm, false);
    for (let k=0;k<m;k++){
      const re = aRe[k]*BwRe[k] - aIm[k]*BwIm[k];
      const im = aRe[k]*BwIm[k] + aIm[k]*BwRe[k];
      aRe[k]=re; aIm[k]=im;
    }
    fftRadix2(aRe, aIm, true);
    for (let k=0;k<n;k++){
      outRe[k] = chirpRe[k]*aRe[k] - chirpIm[k]*aIm[k];
      outIm[k] = chirpRe[k]*aIm[k] + chirpIm[k]*aRe[k];
    }
  }

  function hanningWindow(n){
    const w = new Float64Array(n);
    if (n === 1){ w[0] = 1; return w; }
    for (let t=0;t<n;t++) w[t] = 0.5 - 0.5*Math.cos(2*Math.PI*t/(n-1));
    return w;
  }

  function buildHfAttenuationCurve(N){
    const minCutoff = 0.005;
    const CUTOFF_FRACTION = minCutoff * Math.pow(1/minCutoff, parseFloat(lowpassRange.value)); // fraction of Nyquist where the rolloff's -3dB point sits
    const ORDER = 3;              // filter order: higher = steeper rolloff above cutoff
    const curve = new Float64Array(N);
    for (let k=0;k<N;k++){
      const freqFrac = (N > 1) ? k/(N-1) : 0; // 0..1, maps linearly to 0..Nyquist
      const ratio = freqFrac / CUTOFF_FRACTION;
      curve[k] = 1 / Math.sqrt(1 + Math.pow(ratio, 2*ORDER));
    }
    return curve;
  }

  function caToAudio(X, N, M, hopScale, normalize){
    const n = 2*(N-1);             
    const hop = Math.max(1, Math.floor(n*hopScale));
    const totalLen = hop*(M-1) + n;
    const audio = new Float64Array(totalLen);
    const window = hanningWindow(n);
    const plan = makeBluesteinPlan(n);
    const hfAtten = buildHfAttenuationCurve(N); 
    const full = new Float64Array(n);   // Hermitian-symmetric real spectrum for this row
    const outRe = new Float64Array(n);
    const outIm = new Float64Array(n);

    for (let i=0;i<M;i++){
      const rowOff = i*N;
      for (let k=0;k<N;k++) full[k] = X[rowOff+k] * hfAtten[k];
      for (let k=1;k<=N-2;k++) full[n-k] = X[rowOff+k] * hfAtten[k];

      bluesteinForwardReal(plan, full, outRe, outIm);

      const start = i*hop;
      for (let t=0;t<n;t++) audio[start+t] += (outRe[t]/n)*window[t];
    }

    if (normalize){
      let peak = 0;
      for (let i=0;i<audio.length;i++){ const a = Math.abs(audio[i]); if (a>peak) peak=a; }
      if (peak > 0){
        const g = 0.95/peak;
        for (let i=0;i<audio.length;i++) audio[i] *= g;
      }
    }
    return {audio, blockSize:n, hopSize:hop};
  }

  function encodeWavBlob(samples, sampleRate){
    const numSamples = samples.length;
    const bytesPerSample = 2;
    const blockAlign = bytesPerSample;
    const byteRate = sampleRate*blockAlign;
    const dataSize = numSamples*bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);
    function writeStr(off, str){ for (let i=0;i<str.length;i++) view.setUint8(off+i, str.charCodeAt(i)); }
    writeStr(0, 'RIFF');
    view.setUint32(4, 36+dataSize, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);       // PCM
    view.setUint16(22, 1, true);       // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);      // bits per sample
    writeStr(36, 'data');
    view.setUint32(40, dataSize, true);
    let off = 44;
    for (let i=0;i<numSamples;i++){
      let v = samples[i];
      v = v < -1 ? -1 : (v > 1 ? 1 : v);
      view.setInt16(off, v < 0 ? v*0x8000 : v*0x7FFF, true);
      off += 2;
    }
    return new Blob([buffer], {type:'audio/wav'});
  }


  const $ = id => document.getElementById(id);
  const wLeft = $('wLeft'), wCenter = $('wCenter'), wRight = $('wRight');
  const wLeftVal = $('wLeftVal'), wCenterVal = $('wCenterVal'), wRightVal = $('wRightVal');
  const maskMinInput = $('maskMinInput'), maskMaxInput = $('maskMaxInput');
  const nInput = $('nInput'), mInput = $('mInput');
  const wrapChk = $('wrapChk'), modeSelect = $('modeSelect');
  const activationSelect = $('activationSelect');
  const slopeRange = $('slopeRange'), slopeVal = $('slopeVal');
  const freqRange = $('freqRange'), freqVal = $('freqVal');
  const phaseRange = $('phaseRange'), phaseVal = $('phaseVal');
  const sin2FreqRange = $('sin2FreqRange'), sin2FreqVal = $('sin2FreqVal');
  const sin2PhaseRange = $('sin2PhaseRange'), sin2PhaseVal = $('sin2PhaseVal');
  const sin3FreqRange = $('sin3FreqRange'), sin3FreqVal = $('sin3FreqVal');
  const sin3PhaseRange = $('sin3PhaseRange'), sin3PhaseVal = $('sin3PhaseVal');
  const pRange = $('pRange'), pVal = $('pVal');
  const tanhMultRange = $('tanhMultRange'), tanhMultVal = $('tanhMultVal');
  const absSlopeRange = $('absSlopeRange'), absSlopeVal = $('absSlopeVal');
  const x2CoefRange = $('x2CoefRange'), x2CoefVal = $('x2CoefVal');
  const x3CoefRange = $('x3CoefRange'), x3CoefVal = $('x3CoefVal');
  const muRange = $('muRange'), muVal = $('muVal');
  const sigmaRange = $('sigmaRange'), sigmaVal = $('sigmaVal');
  const seedInput = $('seedInput');
  const seedRow = $('seedRow');
  const initSelect = $('initSelect');
  const initHint = $('initHint');
  const impulsePosRange = $('impulsePosRange'), impulsePosVal = $('impulsePosVal');
  const impulsePosRow = $('impulsePosRow');
  const nCellsInput = $('nCellsInput');
  const nCellsLabel = $('nCellsLabel');
  const cmapSelect = $('cmapSelect');
  const titleInput = $('titleInput');
  const canvas = $('caCanvas');
  const ctx = canvas.getContext('2d');
  const statsEl = $('stats');
  const hopScaleRange = $('hopScaleRange'), hopScaleVal = $('hopScaleVal');
  const normalizeChk = $('normalizeChk');
  const sampleRateInput = $('sampleRateInput');
  const rateFactorSelect = $('rateFactorSelect');
  const durationInput = $('lengthMultInput');
  const genAudioBtn = $('genAudioBtn');
  const lowpassRange = $('lowpassRange'), lowpassVal = $('lowpassVal');
  const audioInfoHint = $('audioInfoHint');
  const audioStatus = $('audioStatus');
  const audioPlayer = $('audioPlayer');
  const savePresetBtn = $('savePresetBtn');
  const loadPresetSelect = $('loadPresetSelect');
  const loadPresetInput = $('loadPresetInput');
  const PRESET_CUSTOM_VALUE = '__custom__';
  const PRESETS_DIR = 'assets/presets/';

  function syncReadout(rangeEl, spanEl, decimals){
    spanEl.textContent = parseFloat(rangeEl.value).toFixed(decimals === undefined ? 2 : decimals);
  }
  const PRECISE_RANGES = [[wLeft,wLeftVal],[wCenter,wCenterVal],[wRight,wRightVal],
   [slopeRange,slopeVal],[freqRange,freqVal],[phaseRange,phaseVal],
   [sin2FreqRange,sin2FreqVal],[sin2PhaseRange,sin2PhaseVal],
   [sin3FreqRange,sin3FreqVal],[sin3PhaseRange,sin3PhaseVal],
   [pRange,pVal],
   [tanhMultRange,tanhMultVal],[absSlopeRange,absSlopeVal],
   [x2CoefRange,x2CoefVal],[x3CoefRange,x3CoefVal],
   [muRange,muVal],[sigmaRange,sigmaVal],[impulsePosRange,impulsePosVal],
   [lowpassRange, lowpassVal]];
  PRECISE_RANGES.forEach(([r,s]) => { syncReadout(r,s,3); });
  syncReadout(hopScaleRange, hopScaleVal);
  PRECISE_RANGES.forEach(([r,s]) => { r.addEventListener('input', () => { syncReadout(r,s,3); scheduleGenerate(); }); });

  const MASK_RANGES = [wLeft, wCenter, wRight];
  function applyMaskBounds(){
    let lo = parseFloat(maskMinInput.value);
    let hi = parseFloat(maskMaxInput.value);
    if (!Number.isFinite(lo)) lo = -3;
    if (!Number.isFinite(hi)) hi = 3;
    if (hi <= lo) hi = lo + 0.001;
    maskMinInput.value = lo;
    maskMaxInput.value = hi;
    MASK_RANGES.forEach(r => {
      r.min = lo; r.max = hi;
      const v = parseFloat(r.value);
      if (v < lo) r.value = lo;
      if (v > hi) r.value = hi;
    });
    syncReadout(wLeft, wLeftVal, 3); syncReadout(wCenter, wCenterVal, 3); syncReadout(wRight, wRightVal, 3);
    generate();
  }
  maskMinInput.addEventListener('change', applyMaskBounds);
  maskMaxInput.addEventListener('change', applyMaskBounds);


  function updateActivationVisibility(){
    const val = activationSelect.value;
    document.querySelectorAll('.subparam').forEach(el => {
      el.classList.toggle('active', el.dataset.for === val);
    });
  }
  activationSelect.addEventListener('change', () => { updateActivationVisibility(); generate(); });
  updateActivationVisibility();

  [wrapChk, modeSelect, cmapSelect].forEach(el => el.addEventListener('change', generate));
  [nInput, mInput].forEach(el => el.addEventListener('change', generate));
  seedInput.addEventListener('change', generate);
  nCellsInput.addEventListener('change', generate);

  function updateInitUI(){
    const val = initSelect.value;
    const isimpulse = val === 'impulse';
    const isRandomN = val === 'random_n';
    seedRow.style.display = isimpulse ? 'none' : 'flex';
    impulsePosRow.style.display = isimpulse ? 'flex' : 'none';
    nCellsLabel.style.display = isRandomN ? '' : 'none';
    nCellsInput.style.display = isRandomN ? '' : 'none';
  }
  initSelect.addEventListener('change', () => { updateInitUI(); generate(); });
  updateInitUI();
  titleInput.addEventListener('input', () => { statsEl.dataset.dirtyTitle = '1'; });

  let lastX = null, lastN = 0, lastM = 0;
  let lastFullResCanvas = null;
  let audioSamples = null;     
  let audioDirty = true;        
  let audioBlobUrl = null;

  function markAudioDirty(){
    audioDirty = true;
    audioStatus.textContent = audioSamples
      ? 'pattern or settings changed — click "Generate audio" to update'
      : 'not generated yet';
  }

  hopScaleRange.addEventListener('input', () => { syncReadout(hopScaleRange, hopScaleVal); markAudioDirty(); });
  normalizeChk.addEventListener('change', markAudioDirty);
  durationInput.addEventListener('change', markAudioDirty);

  function targetLengthMultiplier(){
    const v = parseFloat(durationInput.value);
    const clamped = Number.isFinite(v) ? Math.min(1000, Math.max(0.01, v)) : 1;
    durationInput.value = clamped;
    return clamped;
  }

  function resampleToLength(input, outLen){
    const inLen = input.length;
    const out = new Float64Array(outLen);
    if (outLen <= 0) return out;
    if (inLen === 0) return out;
    if (inLen === 1 || outLen === 1){ out.fill(input[0]); return out; }
    const scale = (inLen - 1) / (outLen - 1);
    for (let i=0;i<outLen;i++){
      const pos = i*scale;
      const i0 = Math.floor(pos);
      const i1 = Math.min(inLen-1, i0+1);
      const frac = pos - i0;
      out[i] = input[i0]*(1-frac) + input[i1]*frac;
    }
    return out;
  }

  function effectiveSampleRate(){
    const base = Math.max(1000, parseInt(sampleRateInput.value,10) || 44100);
    const factor = parseFloat(rateFactorSelect.value);
    return Math.round(base*factor);
  }

  function rebuildWav(){
    if (!audioSamples) return;
    const rate = effectiveSampleRate();
    const blob = encodeWavBlob(audioSamples, rate);
    if (audioBlobUrl) URL.revokeObjectURL(audioBlobUrl);
    audioBlobUrl = URL.createObjectURL(blob);
    audioPlayer.src = audioBlobUrl;
    const dur = audioSamples.length / rate;
    audioStatus.textContent = 'ready — ' + rate + ' Hz, ' + dur.toFixed(2) + 's';
  }

  [sampleRateInput, rateFactorSelect].forEach(el => el.addEventListener('change', () => {
    if (!audioDirty) rebuildWav();
  }));

  function generateAudio(){
    if (!lastX){ audioStatus.textContent = 'generate a pattern first'; return; }
    genAudioBtn.disabled = true;
    genAudioBtn.textContent = 'Computing…';

    setTimeout(() => {
      const hopScale = parseFloat(hopScaleRange.value);
      const normalize = normalizeChk.checked;
      const t0 = performance.now();
      let {audio, blockSize, hopSize} = caToAudio(lastX, lastN, lastM, hopScale, normalize);

      const rawSamples = audio.length;
      const mult = targetLengthMultiplier();
      const targetLen = Math.max(1, Math.round(mult * rawSamples));
      audio = resampleToLength(audio, targetLen);
      if (normalize){
        let peak = 0;
        for (let i=0;i<audio.length;i++){ const a = Math.abs(audio[i]); if (a>peak) peak=a; }
        if (peak > 0){
          const g = 0.95/peak;
          for (let i=0;i<audio.length;i++) audio[i] *= g;
        }
      }

      const t1 = performance.now();
      audioSamples = audio;
      audioDirty = false;
      rebuildWav();
      audioInfoHint.textContent = 'block=' + blockSize + ' hop=' + hopSize +
        ' natural=' + rawSamples + ' × ' + mult.toFixed(2) + ' = ' + audio.length + ' samples' +
        '  (' + (t1-t0).toFixed(0) + 'ms compute)';
      genAudioBtn.disabled = false;
      genAudioBtn.textContent = 'Generate audio';
    }, 10);
  }
  genAudioBtn.addEventListener('click', generateAudio);

  function collectPreset(){
    const cfg = readParams();
    return {
      version: 1,
      title: titleInput.value,
      mask: cfg.mask,
      maskRange: [parseFloat(maskMinInput.value), parseFloat(maskMaxInput.value)],
      N: cfg.N,
      M: cfg.M,
      wrap: cfg.wrap,
      mode: cfg.mode,
      activation: cfg.activation,
      activationParams: cfg.activationParams,
      init: cfg.init,
      seed: cfg.seed,
      impulsePos: cfg.impulsePos,
      nCells: cfg.nCells,
      cmap: cfg.cmap,
      sonification: {
        hopScale: parseFloat(hopScaleRange.value),
        normalize: normalizeChk.checked,
        sampleRate: parseInt(sampleRateInput.value, 10) || 44100,
        rateFactor: parseFloat(rateFactorSelect.value),
        lengthMultiplier: parseFloat(durationInput.value),
        lowpass: parseFloat(lowpassRange.value)
      }
    };
  }

  function savePreset(){
    const preset = collectPreset();
    const blob = new Blob([JSON.stringify(preset, null, 2)], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const name = (titleInput.value || 'ca1d').trim().replace(/[^a-z0-9_-]+/gi,'_').toLowerCase() || 'ca1d';
    const link = document.createElement('a');
    link.href = url;
    link.download = name + '.json';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function deriveNameFromFilename(filename){
    if (!filename) return '';
    return filename.replace(/\.json$/i, '').replace(/$/i, '');
  }

  function applyPreset(p, sourceFilename){
    if (!p || typeof p !== 'object') throw new Error('file is not a JSON object');

    if (Array.isArray(p.maskRange) && p.maskRange.length === 2 && p.maskRange.every(Number.isFinite) && p.maskRange[1] > p.maskRange[0]){
      maskMinInput.value = p.maskRange[0];
      maskMaxInput.value = p.maskRange[1];
      MASK_RANGES.forEach(r => { r.min = p.maskRange[0]; r.max = p.maskRange[1]; });
    }
    if (Array.isArray(p.mask) && p.mask.length === 3 && p.mask.every(Number.isFinite)){
      wLeft.value = p.mask[0]; wCenter.value = p.mask[1]; wRight.value = p.mask[2];
    }
    if (Number.isFinite(p.N)) nInput.value = p.N;
    if (Number.isFinite(p.M)) mInput.value = p.M;
    if (typeof p.wrap === 'boolean') wrapChk.checked = p.wrap;
    if (p.mode === 'mod' || p.mode === 'clip') modeSelect.value = p.mode;
    const activationCompat = p.activation === 'identity' ? 'linear' : p.activation;
    if (typeof activationCompat === 'string' && activationSelect.querySelector('option[value="'+activationCompat+'"]')) {
      activationSelect.value = activationCompat;
    }
    const ap = p.activationParams || {};
    if (Number.isFinite(ap.slope)) slopeRange.value = ap.slope;
    if (Number.isFinite(ap.freq)) freqRange.value = ap.freq;
    if (Number.isFinite(ap.phase)) phaseRange.value = ap.phase;
    if (Number.isFinite(ap.sin2Freq)) sin2FreqRange.value = ap.sin2Freq;
    if (Number.isFinite(ap.sin2Phase)) sin2PhaseRange.value = ap.sin2Phase;
    if (Number.isFinite(ap.sin3Freq)) sin3FreqRange.value = ap.sin3Freq;
    if (Number.isFinite(ap.sin3Phase)) sin3PhaseRange.value = ap.sin3Phase;
    if (Number.isFinite(ap.p)) pRange.value = ap.p;
    if (Number.isFinite(ap.tanhMult)) tanhMultRange.value = ap.tanhMult;
    if (Number.isFinite(ap.absSlope)) absSlopeRange.value = ap.absSlope;
    if (Number.isFinite(ap.x2Coef)) x2CoefRange.value = ap.x2Coef;
    if (Number.isFinite(ap.x3Coef)) x3CoefRange.value = ap.x3Coef;
    if (Number.isFinite(ap.mu)) muRange.value = ap.mu;
    if (Number.isFinite(ap.sigma)) sigmaRange.value = ap.sigma;
    if (p.init === 'random' || p.init === 'impulse' || p.init === 'random_n') initSelect.value = p.init;
    if (p.seed === null || p.seed === undefined || p.seed === '') seedInput.value = '';
    else if (Number.isFinite(p.seed)) seedInput.value = p.seed;
    if (Number.isFinite(p.impulsePos)) impulsePosRange.value = Math.min(1, Math.max(0, p.impulsePos));
    if (Number.isFinite(p.nCells)) nCellsInput.value = Math.max(1, Math.round(p.nCells));
    if (p.cmap === 'binary' || p.cmap === 'gray') cmapSelect.value = p.cmap;
    const filenameTitle = deriveNameFromFilename(sourceFilename).trim();
    const resolvedTitle = filenameTitle !== ''
      ? filenameTitle
      : (typeof p.title === 'string' ? p.title.trim() : '');
    if (resolvedTitle) titleInput.value = resolvedTitle;

    const son = p.sonification || {};
    if (Number.isFinite(son.hopScale)) hopScaleRange.value = son.hopScale;
    if (typeof son.normalize === 'boolean') normalizeChk.checked = son.normalize;
    if (Number.isFinite(son.sampleRate)) sampleRateInput.value = son.sampleRate;
    if (son.rateFactor !== undefined && rateFactorSelect.querySelector('option[value="'+son.rateFactor+'"]')){
      rateFactorSelect.value = String(son.rateFactor);
    }
    if (Number.isFinite(son.lengthMultiplier)) durationInput.value = son.lengthMultiplier;
    if (Number.isFinite(son.lowpass)) lowpassRange.value = son.lowpass;

    PRECISE_RANGES.forEach(([r,s]) => syncReadout(r,s,3));
    syncReadout(hopScaleRange, hopScaleVal);

    updateActivationVisibility();
    updateInitUI();
    refreshCselTriggers();
    generate();
    generateAudio();
  }

  function loadPresetFile(file){
    const reader = new FileReader();
    reader.onload = () => {
      try {
        applyPreset(JSON.parse(reader.result), file.name);
      } catch (err){
      }
    };
    reader.readAsText(file);
  }

  savePresetBtn.addEventListener('click', savePreset);

  async function fetchPresetManifest(){
    try {
      const res = await fetch(PRESETS_DIR + 'manifest.json', {cache:'no-store'});
      if (res.ok){
        const data = await res.json();
        if (Array.isArray(data)){
          return data.map(entry => {
            if (typeof entry === 'string') return {file: entry, label: deriveNameFromFilename(entry)};
            if (entry && typeof entry === 'object' && entry.file) return {file: entry.file, label: entry.label || deriveNameFromFilename(entry.file)};
            return null;
          }).filter(Boolean);
        }
      }
    } catch (err){ /* fall through to directory-listing fallback */ }

    try {
      const res = await fetch(PRESETS_DIR, {cache:'no-store'});
      if (res.ok){
        const html = await res.text();
        const doc = new DOMParser().parseFromString(html, 'text/html');
        const files = Array.from(doc.querySelectorAll('a[href$=".json"]'))
          .map(a => decodeURIComponent(a.getAttribute('href')))
          .filter(href => !href.includes('/'));
        return files.map(file => ({file, label: deriveNameFromFilename(file)}));
      }
    } catch (err){ /* no manifest and no listable directory */ }

    return [];
  }

  async function populatePresetSelect(){
    const entries = await fetchPresetManifest();
    while (loadPresetSelect.options.length > 2) loadPresetSelect.remove(2);
    entries.forEach(({file, label}) => {
      const opt = document.createElement('option');
      opt.value = file;
      opt.textContent = label;
      loadPresetSelect.appendChild(opt);
    });
  }

  function loadPresetFromFolder(file){
    fetch(PRESETS_DIR + file, {cache:'no-store'})
      .then(res => {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(json => applyPreset(json, file))
      .catch(err => {
        console.error('Failed to load preset', file, err);
        alert('Could not load preset "' + file + '": ' + err.message);
      });
  }

  loadPresetSelect.addEventListener('change', () => {
    const val = loadPresetSelect.value;
    if (val === PRESET_CUSTOM_VALUE){
      loadPresetInput.click();
    } else if (val) {
      loadPresetFromFolder(val);
    }
    loadPresetSelect.value = '';
  });

  loadPresetInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) loadPresetFile(file);
    e.target.value = '';
  });

  populatePresetSelect();

  const openCselPanels = [];
  const cselRegistry = [];
  function closeAllCsel(except){
    openCselPanels.slice().forEach(p => { if (p !== except) p.close(); });
  }

  function refreshCselTriggers(){
    cselRegistry.forEach(r => r.sync());
  }

  function enhanceSelect(selectEl, extraTriggerClass){
    const wrap = document.createElement('div');
    wrap.className = 'csel-wrap';
    selectEl.parentNode.insertBefore(wrap, selectEl);
    wrap.appendChild(selectEl);
    selectEl.classList.add('csel-native');
    selectEl.tabIndex = -1;

    const trigger = document.createElement('div');
    trigger.className = 'csel-trigger' + (extraTriggerClass ? ' ' + extraTriggerClass : '');
    trigger.tabIndex = 0;
    wrap.appendChild(trigger);

    const panel = document.createElement('div');
    panel.className = 'csel-panel';
    wrap.appendChild(panel);

    function syncTrigger(){
      const opt = selectEl.options[selectEl.selectedIndex];
      trigger.textContent = opt ? opt.textContent : '';
    }

    function buildPanel(){
      panel.innerHTML = '';
      Array.from(selectEl.options).forEach(opt => {
        if (opt.hidden) return;
        const item = document.createElement('div');
        item.className = 'csel-option' +
          (opt.value === selectEl.value ? ' csel-selected' : '') +
          (opt.disabled ? ' csel-disabled' : '');
        item.textContent = opt.textContent;
        item.addEventListener('click', () => {
          if (opt.disabled) return;
          selectEl.value = opt.value;
          selectEl.dispatchEvent(new Event('change', {bubbles:true}));
          syncTrigger();
          api.close();
        });
        panel.appendChild(item);
      });
    }

    function positionPanel(){
      const rect = trigger.getBoundingClientRect();
      const vh = window.innerHeight;
      const gap = 2, edgeMargin = 8, minH = 80, desiredMax = 240;
      const spaceBelow = vh - rect.bottom - gap - edgeMargin;
      const spaceAbove = rect.top - gap - edgeMargin;

      panel.style.left = Math.round(rect.left) + 'px';
      panel.style.width = Math.round(rect.width) + 'px';

      if (spaceBelow >= minH || spaceBelow >= spaceAbove){
        panel.style.top = Math.round(rect.bottom + gap) + 'px';
        panel.style.bottom = 'auto';
        panel.style.maxHeight = Math.max(minH, Math.min(desiredMax, spaceBelow)) + 'px';
      } else {
        panel.style.bottom = Math.round(vh - rect.top + gap) + 'px';
        panel.style.top = 'auto';
        panel.style.maxHeight = Math.max(minH, Math.min(desiredMax, spaceAbove)) + 'px';
      }
    }
    function onViewportChange(){ positionPanel(); }

    function onDocDown(e){ if (!wrap.contains(e.target)) api.close(); }

    function open(){
      closeAllCsel(api);
      buildPanel();
      positionPanel();
      panel.classList.add('csel-open');
      trigger.classList.add('csel-open');
      document.addEventListener('mousedown', onDocDown, true);
      window.addEventListener('scroll', onViewportChange, true);
      window.addEventListener('resize', onViewportChange, true);
      if (!openCselPanels.includes(api)) openCselPanels.push(api);
    }
    function close(){
      panel.classList.remove('csel-open');
      trigger.classList.remove('csel-open');
      document.removeEventListener('mousedown', onDocDown, true);
      window.removeEventListener('scroll', onViewportChange, true);
      window.removeEventListener('resize', onViewportChange, true);
      const idx = openCselPanels.indexOf(api);
      if (idx !== -1) openCselPanels.splice(idx, 1);
    }

    trigger.addEventListener('click', () => {
      if (panel.classList.contains('csel-open')) api.close(); else api.open();
    });
    trigger.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' '){ e.preventDefault(); trigger.click(); }
      else if (e.key === 'Escape'){ api.close(); }
    });

    selectEl.addEventListener('change', syncTrigger);
    syncTrigger();

    const api = { open, close };
    cselRegistry.push({ sync: syncTrigger });
    return api;
  }

  enhanceSelect(loadPresetSelect, 'csel-trigger--preset');
  enhanceSelect(initSelect);
  enhanceSelect(activationSelect);
  enhanceSelect(modeSelect);

  function randomInRange(min, max){ return min + Math.random()*(max-min); }

  $('randomizeBtn').addEventListener('click', () => {
    wLeft.value   = randomInRange(parseFloat(wLeft.min), parseFloat(wLeft.max)).toFixed(3);
    wCenter.value = randomInRange(parseFloat(wCenter.min), parseFloat(wCenter.max)).toFixed(3);
    wRight.value  = randomInRange(parseFloat(wRight.min), parseFloat(wRight.max)).toFixed(3);

    const activations = Array.from(activationSelect.options).map(o => o.value);
    activationSelect.value = activations[Math.floor(Math.random()*activations.length)];

    slopeRange.value    = randomInRange(parseFloat(slopeRange.min), parseFloat(slopeRange.max)).toFixed(3);
    freqRange.value     = randomInRange(parseFloat(freqRange.min), parseFloat(freqRange.max)).toFixed(3);
    phaseRange.value    = randomInRange(parseFloat(phaseRange.min), parseFloat(phaseRange.max)).toFixed(3);
    sin2FreqRange.value  = randomInRange(parseFloat(sin2FreqRange.min), parseFloat(sin2FreqRange.max)).toFixed(3);
    sin2PhaseRange.value = randomInRange(parseFloat(sin2PhaseRange.min), parseFloat(sin2PhaseRange.max)).toFixed(3);
    sin3FreqRange.value  = randomInRange(parseFloat(sin3FreqRange.min), parseFloat(sin3FreqRange.max)).toFixed(3);
    sin3PhaseRange.value = randomInRange(parseFloat(sin3PhaseRange.min), parseFloat(sin3PhaseRange.max)).toFixed(3);
    pRange.value        = randomInRange(parseFloat(pRange.min), parseFloat(pRange.max)).toFixed(3);
    tanhMultRange.value = randomInRange(parseFloat(tanhMultRange.min), parseFloat(tanhMultRange.max)).toFixed(3);
    absSlopeRange.value = randomInRange(parseFloat(absSlopeRange.min), parseFloat(absSlopeRange.max)).toFixed(3);
    x2CoefRange.value = randomInRange(parseFloat(x2CoefRange.min), parseFloat(x2CoefRange.max)).toFixed(3);
    x3CoefRange.value = randomInRange(parseFloat(x3CoefRange.min), parseFloat(x3CoefRange.max)).toFixed(3);
    muRange.value    = randomInRange(parseFloat(muRange.min), parseFloat(muRange.max)).toFixed(3);
    sigmaRange.value = randomInRange(parseFloat(sigmaRange.min), parseFloat(sigmaRange.max)).toFixed(3);

    wrapChk.checked = Math.random() < 0.5;
    modeSelect.value = Math.random() < 0.5 ? 'mod' : 'clip';

    PRECISE_RANGES.forEach(([r,s]) => syncReadout(r,s,3));
    updateActivationVisibility();
    refreshCselTriggers();
    generate();
  });
  $('randSeedBtn').addEventListener('click', () => {
    seedInput.value = Math.floor(Math.random()*1e6);
    generate();
  });
  $('saveBtn').addEventListener('click', savePNG);

  let debounceTimer = null;
  function scheduleGenerate(){
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(generate, 30);
  }

  function readParams(){
    let N = Math.max(10, Math.min(2048, parseInt(nInput.value,10) || 220));
    let M = Math.max(10, Math.min(10000, parseInt(mInput.value,10) || 220));
    nInput.value = N; mInput.value = M;

    const seedRaw = seedInput.value.trim === undefined ? seedInput.value : seedInput.value;
    const seed = seedInput.value === '' ? null : parseInt(seedInput.value, 10);

    return {
      N, M, seed,
      wrap: wrapChk.checked,
      mode: modeSelect.value,
      activation: activationSelect.value,
      activationParams: {
        slope: parseFloat(slopeRange.value),
        freq: parseFloat(freqRange.value),
        phase: parseFloat(phaseRange.value),
        sin2Freq: parseFloat(sin2FreqRange.value),
        sin2Phase: parseFloat(sin2PhaseRange.value),
        sin3Freq: parseFloat(sin3FreqRange.value),
        sin3Phase: parseFloat(sin3PhaseRange.value),
        p: parseFloat(pRange.value),
        tanhMult: parseFloat(tanhMultRange.value),
        absSlope: parseFloat(absSlopeRange.value),
        x2Coef: parseFloat(x2CoefRange.value),
        x3Coef: parseFloat(x3CoefRange.value),
        mu: parseFloat(muRange.value),
        sigma: parseFloat(sigmaRange.value)
      },
      mask: [parseFloat(wLeft.value), parseFloat(wCenter.value), parseFloat(wRight.value)],
      cmap: cmapSelect.value,
      init: initSelect.value,
      impulsePos: parseFloat(impulsePosRange.value),
      nCells: Math.min(N, Math.max(1, parseInt(nCellsInput.value, 10) || 1)),
      lowpass: parseFloat(lowpassRange.value)
    };
  }

  function render(X, N, M, cmap){
    const off = document.createElement('canvas');
    off.width = N; off.height = M;
    const offCtx = off.getContext('2d');
    const imgData = offCtx.createImageData(N, M);
    const data = imgData.data;

    for (let idx=0, px=0; idx<X.length; idx++, px+=4){
      const v = X[idx];
      let gray;
      if (cmap === 'binary') gray = Math.round(255 * (1 - v));
      else gray = Math.round(255 * v);
      data[px] = gray; data[px+1] = gray; data[px+2] = gray; data[px+3] = 255;
    }
    offCtx.putImageData(imgData, 0, 0);
    lastFullResCanvas = off;

    const maxBox = 600;
    const scale = Math.max(1, Math.min(maxBox / N, maxBox / M));
    // const scale = Math.max(1, maxBox / N);
    // const scale = maxBox / N;
    canvas.width = Math.round(N * scale);
    canvas.height = Math.round(M * scale);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0,0,canvas.width, canvas.height);
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  }

  function generate(){
    const cfg = readParams();
    const t0 = performance.now();
    const X = generateContinuousCA(cfg);
    const t1 = performance.now();
    render(X, cfg.N, cfg.M, cfg.cmap);
    const t2 = performance.now();
    lastX = X; lastN = cfg.N; lastM = cfg.M;
    markAudioDirty();
    statsEl.textContent =
      'N=' + cfg.N + '  M=' + cfg.M +
      '  mask=[' + cfg.mask.map(v=>v.toFixed(3)).join(', ') + ']' +
      '  activation=' + cfg.activation +
      '  mode=' + cfg.mode +
      '  wrap=' + cfg.wrap +
      '  init=' + cfg.init +
      (cfg.init === 'impulse' ? ('  pos=' + cfg.impulsePos.toFixed(3))
        : cfg.init === 'random_n' ? ('  n=' + cfg.nCells + '  seed=' + (cfg.seed===null ? 'random' : cfg.seed))
        : ('  seed=' + (cfg.seed===null ? 'random' : cfg.seed)));
  }

  function savePNG(){
    if (!lastFullResCanvas) return;
    const link = document.createElement('a');
    const name = (titleInput.value || 'ca1d').trim().replace(/[^a-z0-9_-]+/gi,'_').toLowerCase() || 'ca1d';
    link.download = name + '_' + lastN + 'x' + lastM + '.png';
    link.href = lastFullResCanvas.toDataURL('image/png'); // exact pixel data, PNG is lossless
    link.click();
  }

  const helpBtn = $('helpBtn');
  const helpOverlay = $('helpOverlay');
  const helpClose = $('helpClose');
  function openHelp(){ helpOverlay.classList.add('open'); }
  function closeHelp(){ helpOverlay.classList.remove('open'); }
  helpBtn.addEventListener('click', openHelp);
  helpClose.addEventListener('click', closeHelp);

  const lightModeChk = $('lightModeChk');
  function applylightMode(){
    document.body.classList.toggle('light-mode', lightModeChk.unchecked);
  }
  lightModeChk.addEventListener('change', applylightMode);


  helpOverlay.addEventListener('mousedown', (e) => { if (e.target === helpOverlay) closeHelp(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeHelp(); });

  generate();
  generateAudio();
  openHelp();
})();
