/* Flowing aurora behind the page. Original shader, in the spirit of a React Bits background. */
(function () {
  const canvas = document.querySelector(".aurora");
  if (!canvas) return;

  const gl = canvas.getContext("webgl", { alpha: false, antialias: false });
  if (!gl) return;

  const vertexSource = "attribute vec2 a;void main(){gl_Position=vec4(a,0.0,1.0);}";
  const fragmentSource = [
    "precision mediump float;",
    "uniform vec2 uRes;",
    "uniform float uTime;",
    "uniform vec2 uPointer;",
    "void main(){",
    "  vec2 uv=gl_FragCoord.xy/uRes;",
    "  float t=uTime;",
    "  float px=(uPointer.x-0.5);",
    "  float py=(uPointer.y-0.5);",
    "  float x=uv.x+px*0.14;",
    "  float y=uv.y;",
    "  float w1=0.78+0.055*sin(x*5.2+t*0.6)+0.025*sin(x*12.0-t*0.35)+px*0.07;",
    "  float w2=0.92+0.04*sin(x*3.1-t*0.42+1.6)+py*0.035;",
    "  float w3=0.62+0.05*sin(x*4.0+t*0.3+0.7)-px*0.05;",
    "  float b1=exp(-pow((y-w1)*10.0,2.0));",
    "  float b2=exp(-pow((y-w2)*8.0,2.0));",
    "  float b3=exp(-pow((y-w3)*7.0,2.0));",
    "  vec3 paper=vec3(0.969,0.961,0.945);",
    "  vec3 gold=vec3(0.97,0.58,0.14);",
    "  vec3 mint=vec3(0.38,0.86,0.58);",
    "  vec3 forest=vec3(0.08,0.42,0.30);",
    "  vec3 color=paper;",
    "  color=mix(color,forest,b3*0.92);",
    "  color=mix(color,mint,b2*0.95);",
    "  color=mix(color,gold,b1*0.94);",
    "  float crown=smoothstep(0.74,1.0,y);",
    "  vec3 crownTint=mix(mint,gold,0.42+0.38*sin(x*2.2+t*0.22));",
    "  color=mix(color,crownTint,crown*0.42);",
    "  float keep=smoothstep(0.18,0.58,y);",
    "  color=mix(paper,color,keep);",
    "  gl_FragColor=vec4(color,1.0);",
    "}"
  ].join("");

  function compile(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    return shader;
  }

  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return;
  gl.useProgram(program);

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, "a");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

  const resolution = gl.getUniformLocation(program, "uRes");
  const time = gl.getUniformLocation(program, "uTime");
  const pointer = gl.getUniformLocation(program, "uPointer");
  const mouse = [0.5, 0.72];
  let frame = 0;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 1.5);
    const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      gl.viewport(0, 0, width, height);
    }
  }

  function draw(now) {
    resize();
    gl.uniform2f(resolution, canvas.width, canvas.height);
    gl.uniform1f(time, reduced ? 1.2 : now * 0.001);
    gl.uniform2f(pointer, mouse[0], mouse[1]);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    if (!reduced && !document.hidden) frame = requestAnimationFrame(draw);
  }

  window.addEventListener("pointermove", (event) => {
    mouse[0] = event.clientX / Math.max(window.innerWidth, 1);
    mouse[1] = 1 - event.clientY / Math.max(window.innerHeight, 1);
  }, { passive: true });
  window.addEventListener("resize", resize);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && !reduced) frame = requestAnimationFrame(draw);
  });
  frame = requestAnimationFrame(draw);
})();
