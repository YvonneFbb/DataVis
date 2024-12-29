let canvas_quad = 1000
let char_quad = 60
let gate = 120

let sampler
let bepoints
let p5bezier
let todo = []

class Sampler {
  constructor (sampleInterval = 300) {
    // Setup the sound inputs
    this.mic = new p5.AudioIn()
    this.fft = new p5.FFT()
    this.fft.setInput(this.mic)

    this.sampleInterval = sampleInterval
    this.lastSampleTime = 0

    this.loudness = 0
    this.centroid = 0

    this.active = false

    this.mic.start()
  }

  // Sample if the time has come
  sample () {
    let currentTime = millis()

    if (currentTime - this.lastSampleTime >= this.sampleInterval) {
      this.lastSampleTime = currentTime

      let spec = this.fft.analyze()

      this.loudness = this.fft.getEnergy('lowMid', 'highMid')
      this.centroid = this.fft.getCentroid()

      if (this.loudness > gate) {
        this.active = true
      }

      // drawSpectrum(spec)
    }

    return {
      loudness: this.loudness,
      centroid: this.centroid
    }
  }

  active () {
    return this.active
  }
}

class BePoints {
  constructor () {
    this.points = []
    this.interpolatedPoints = []
    this.index = 0
    this.size = 200

    this.x = 0
    this.y = canvas_quad / 2
  }

  addBlock (coordinates) {
    // Points are [0~60, 0~60] squares
    for (let i = 0; i < coordinates.length; i++) {
      let [x, y] = coordinates[i]
      x = this.x + (x / char_quad) * this.size
      y = this.y + (y / char_quad - 0.5) * this.size
      // console.log('add points: ', x, y)
      this.points.push(createVector(x, y))
    }

    this.x += this.size
  }

  interpolatePoints () {
    if (this.points.length < 2) return

    for (let i = 0; i < this.points.length - 1; i++) {
      let start = this.points[i]
      let end = this.points[i + 1]

      let numNewPoints = 5
      for (let j = 0; j <= numNewPoints; j++) {
        let t = j / (numNewPoints + 1)
        let interpolatedPoint = p5.Vector.lerp(start, end, t)
        // console.log('interpolated points: ', interpolatedPoint)
        this.interpolatedPoints.push(interpolatedPoint)
      }
    }

    console.log(this.interpolatedPoints)
  }

  pop (loudness, centroid) {
    let skip = floor(map(centroid, 0, 8000, 5, 1))
    skip = skip < 1 ? 1 : skip

    let scaleX
    let scaleY
    if (loudness < gate) {
      scaleX = map(loudness, 0, gate, 0.6, 0.8)
      scaleY = map(loudness, 0, gate, 0, 0.6)
    } else {
      scaleX = map(loudness, gate, 255, 0.8, 1.25)
      scaleY = map(loudness, gate, 255, 0.6, 1.25)
    }
    // console.log(loudness, centroid, skip, scaleX, scaleY)

    if (this.index + skip >= this.interpolatedPoints.length) {
      return null
    }

    let p = this.interpolatedPoints[this.index]
    this.index += skip

    // Processing points
    p.x *= scaleX
    p.y *= scaleY

    return p
  }
}

function setup () {
  // Setup the canvas.
  const c = createCanvas(canvas_quad, canvas_quad)
  background(255)

  sampler = new Sampler(300)
  bepoints = new BePoints()

  bepoints.addBlock(char_data.q)
  bepoints.addBlock(char_data.i)
  bepoints.addBlock(char_data.n)
  bepoints.addBlock(char_data.g)
  bepoints.interpolatePoints()

  p5bezier = initBezier(c)
}

function draw () {
  let s = sampler.sample()
  // console.log(s)

  if (sampler.active) {
    // Ok, here we do our jobs
    let p = bepoints.pop(s.loudness, s.centroid)

    if (p) {
      background(255, 15)
      todo.push([p.x, p.y])

      push()
      noFill()
      stroke(0)
      strokeWeight(3)
      if (todo.length >= 2) p5bezier.draw(todo)
      pop()

      // for (let i = 0; i < todo.length; i++) {
      //   let p = todo[i]
      //   fill(255, 0, 0)
      //   ellipse(p[0], p[1], 5, 5)
      // }
    }
  }
}

function drawSpectrum (spectrum) {
  background(0)
  stroke(255)
  noFill()
  beginShape()
  for (let i = 0; i < spectrum.length; i++) {
    let x = map(i, 0, spectrum.length, 0, width)
    let y = map(spectrum[i], 0, 255, height, 0)
    vertex(x, y)
  }
  endShape()
}

// Store our datas
char_data = {
  g: [
    [0, 30],
    [5.28, 30.827],
    [9.563, 31.327],
    [19.769, 32.043],
    [29.937, 31.043],
    [34.471, 28.418],
    [33.351, 23.305],
    [31.33, 22.357],
    [27.152, 22.598],
    [23.204, 25.953],
    [22.395, 31.362],
    [24.205, 36.63],
    [26.68, 40.2],
    [29.519, 43.834],
    [30.112, 45.666],
    [29.612, 47.669],
    [26.768, 52.441],
    [22.395, 56.666],
    [18.62, 58.302],
    [17.184, 57.194],
    [16.684, 55.666],
    [17.62, 51.604],
    [19.988, 47.96],
    [22.895, 44.834],
    [27.487, 41.122],
    [31.83, 37.954],
    [35.516, 35.843],
    [39.771, 33.798],
    [47.401, 31.427],
    [54.778, 30.205],
    [60, 30]
  ],
  i: [
    [0, 30],
    [5.408, 29.966],
    [10.987, 29.248],
    [14.47, 28.488],
    [17.084, 27.646],
    [20.016, 26.413],
    [22.644, 24.682],
    [24.025, 23.305],
    [25.662, 20.991],
    [26.557, 18.547],
    [26.94, 15.566],
    [26.759, 13.229],
    [26.327, 10.406],
    [26.004, 7.978],
    [25.662, 5.265],
    [24.987, 2.523],
    [24.225, 8.84],
    [24.044, 12.178],
    [23.924, 15.895],
    [24.025, 19.6],
    [24.298, 27.312],
    [24.555, 29.973],
    [24.835, 33.842],
    [25.229, 36.63],
    [25.687, 39.299],
    [26.558, 41.159],
    [27.752, 38.895],
    [28.448, 36.032],
    [30.276, 33.716],
    [32.999, 32.144],
    [36.062, 31.037],
    [38.38, 30.537],
    [41.451, 29.973],
    [45.551, 29.973],
    [60, 30]
  ],
  n: [
    [0, 30],
    [9.243, 25.957],
    [15.88, 22.554],
    [22.848, 17.701],
    [26.68, 12.572],
    [25.948, 8.735],
    [25.448, 16.138],
    [25.345, 20.47],
    [25.38, 28.22],
    [24.948, 38.279],
    [24.448, 45.067],
    [22.848, 49.117],
    [22.246, 44.567],
    [22.848, 37.279],
    [23.448, 32.454],
    [24.637, 27.134],
    [26.448, 22.054],
    [28.952, 17.701],
    [34.511, 17.201],
    [37.147, 22.054],
    [36.192, 28.173],
    [35.318, 33.847],
    [36.493, 37.893],
    [41.456, 38.13],
    [44.467, 35.513],
    [47.371, 33.483],
    [51.46, 31.415],
    [55.643, 30.241],
    [60, 30]
  ],

  q: [
    [0, 30],
    [5.919, 29.815],
    [8.961, 29.533],
    [19.88, 26.624],
    [24.57, 24.269],
    [30.101, 21.255],
    [34.425, 17.365],
    [36.301, 10.97],
    [31.625, 3.868],
    [26.533, 5.069],
    [21.725, 11.082],
    [20.883, 18.347],
    [23.094, 25.063],
    [24.772, 30.205],
    [26.615, 35.998],
    [28.82, 42.632],
    [31.946, 51.586],
    [32.312, 54.449],
    [30.311, 56.602],
    [29.021, 55.12],
    [28.51, 50.814],
    [28.51, 46.163],
    [30.364, 40.988],
    [35.784, 35.99],
    [46.944, 31.618],
    [52.286, 30.558],
    [60, 30]
  ]
}

line_data = {
  horizontal: [
    [0, 30],
    [15, 30],
    [30, 30],
    [45, 30],
    [60, 30]
  ]
}
