let canvas_quad = 1000
let char_quad = 60
let gate = 30
let drawType = 0
let debugPoint = 0

let sampler
let bepoints
let p5bezier
let todo = []
let bgImage

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
    }

    // return {
    //   loudness: this.loudness,
    //   centroid: this.centroid
    // }

    return {
      loudness: 50,
      centroid: 1000
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
    this.blockIndex = 0
    this.pointIndex = 0
    this.xsize = 70
    this.ysize = 150

    this.x = 75
    this.y = canvas_quad / 2 - 50
  }

  addBlock (coordinates) {
    // Points are [0~60, 0~60] squares
    let block = []
    for (let i = 0; i < coordinates.length; i++) {
      let [x, y] = coordinates[i]
      x = this.x + (x / char_quad) * this.xsize
      y = this.y + (y / char_quad - 0.5) * this.ysize
      // console.log('add points: ', x, y)
      block.push(createVector(x, y))
    }
    this.points.push(block)
    this.x += this.xsize
  }

  interpolatePoints () {
    for (let block of this.points) {
      let interpolateBlocks = []
      for (let i = 0; i < block.length - 1; i++) {
        let start = block[i]
        let end = block[i + 1]

        let numNewPoints = 2
        for (let j = 0; j <= numNewPoints; j++) {
          let t = j / (numNewPoints + 1)
          let interpolatedPoint = p5.Vector.lerp(start, end, t)
          interpolateBlocks.push(interpolatedPoint)
          // console.log('interpolated points: ', interpolatedPoint)
        }
      }
      this.interpolatedPoints.push(interpolateBlocks)
    }

    console.log(this.interpolatedPoints)
  }

  pop (loudness, centroid) {
    let skip = floor(map(centroid, 2000, 4000, 6, 1))
    skip = skip < 1 ? 1 : skip

    let scaleX
    let scaleY
    if (loudness < gate) {
      // scaleX = map(loudness, 0, gate, 0.6, 0.8)
      scaleY = map(loudness, 0, gate, 0, 0.6)
    } else {
      // scaleX = map(loudness, gate, 255, 0.8, 1.25)
      scaleY = map(loudness, gate, 150, 0.75, 1.5)
    }
    // console.log(loudness, centroid, skip, scaleY)

    if (this.blockIndex >= this.interpolatedPoints.length) {
      return null
    }
    if (this.pointIndex >= this.interpolatedPoints[this.blockIndex].length) {
      this.pointIndex = 0
      this.blockIndex += 1
    }
    if (this.blockIndex >= this.interpolatedPoints.length) {
      return null
    }

    let p = this.interpolatedPoints[this.blockIndex][this.pointIndex]
    this.pointIndex += skip

    // Processing points
    // p.x *= scaleX
    p.y = (p.y - this.y) * scaleY + this.y

    return {
      block: this.blockIndex + 1,
      point: [p.x, p.y]
    }
  }
}

function setup () {
  // Setup the canvas.
  const c = createCanvas(canvas_quad, canvas_quad)
  background(255)
  bgImage = loadImage('image.png')

  sampler = new Sampler(300)
  bepoints = new BePoints()

  // Init lines
  switch (drawType) {
    case 0:
      {
        let eblock = []
        for (let i = 0; i < 4; i++) {
          eblock.push([bepoints.x, bepoints.y])
          bepoints.x += 5
        }
        todo.push(eblock)
      }
      break
    case 1:
      {
        for (let i = 0; i < 4; i++) {
          todo.push([bepoints.x, bepoints.y])
          bepoints.x += 5
        }
      }
      break
  }

  bepoints.addBlock(char_data.q)
  bepoints.addBlock(char_data.i)
  bepoints.addBlock(char_data.n)
  bepoints.addBlock(char_data.g)
  bepoints.interpolatePoints()

  p5bezier = initBezier(c)
}

function draw () {
  let s = sampler.sample()

  if (sampler.active) {
    switch (drawType) {
      case 0:
        {
          // Ok, here we do our jobs
          let p = bepoints.pop(s.loudness, s.centroid)
          // console.log(p, todo)
          if (p) {
            background(255, 30)
            image(bgImage, 0, canvas_quad / 3, canvas_quad / 3, canvas_quad / 3)
            if (p.block >= todo.length) {
              todo.push([])
              todo[p.block].push(todo[p.block - 1].at(-2))
              todo[p.block].push(todo[p.block - 1].at(-1))
            }
            todo[p.block].push(p.point)
            push()
            noFill()
            stroke(0)
            strokeWeight(4)
            for (let block of todo) {
              p5bezier.draw(block, 'OPEN', 2)
            }
            pop()

            // console.log(todo)
            if (debugPoint) {
              for (let block of todo) {
                for (let i = 0; i < block.length; i++) {
                  let p = block[i]
                  fill(255, 0, 0)
                  ellipse(p[0], p[1], 5, 5)
                }
              }
            }
          }
        }
        break
      case 1: {
        let p = bepoints.pop(s.loudness, s.centroid)
        if (p) {
          background(255, 30)
          image(bgImage, 0, canvas_quad / 3, canvas_quad / 3, canvas_quad / 3)

          todo.push(p.point)
          push()
          noFill()
          stroke(0)
          strokeWeight(4)
          p5bezier.draw(todo, 'OPEN', 2)
          pop()

          if (debugPoint) {
            for (let i = 0; i < todo.length; i++) {
              let p = todo[i]
              fill(255, 0, 0)
              ellipse(p[0], p[1], 5, 5)
            }
          }
        }
      }
    }

    // {
    //   let all_points = bepoints.all(s.loudness, s.centroid)
    //   background(255, 15)
    //   push()
    //   noFill()
    //   stroke(0)
    //   strokeWeight(3)
    //   if (all_points.length >= 2) p5bezier.draw(all_points)
    //   pop()
    //   // for (let i = 0; i < all_points.length; i++) {
    //   //   let p = all_points[i]
    //   //   fill(255, 0, 0)
    //   //   ellipse(p[0], p[1], 5, 5)
    //   // }
    // }
  }
}

function keyPressed () {
  if (key === 's' || key === 'S') {
    let pg = createGraphics(canvas_quad, canvas_quad)
    pg.clear()
    pg.image(get(), 0, 0)

    pg.loadPixels()
    for (let y = 0; y < canvas_quad * 4; y++) {
      for (let x = 0; x < canvas_quad; x++) {
        let index = (x + y * canvas_quad) * 4
        let r = pg.pixels[index]
        let g = pg.pixels[index + 1]
        let b = pg.pixels[index + 2]
        let a = pg.pixels[index + 3]

        if (r >= 250 && g >= 250 && b >= 250) {
          pg.pixels[index + 3] = 0
        }
      }
      console.log(y)
    }
    pg.updatePixels()

    save(pg, 'canvas.png')
  }
}

// Store our datas
char_data = {
  a: [
    [0, 30],
    [6.053, 30.892],
    [13.371, 30.611],
    [20.035, 29.778],
    [25.459, 28.739],
    [31.553, 27.262],
    [35.851, 25.901],
    [40.186, 23.988],
    [40.834, 22.026],
    [38.215, 20.313],
    [35.851, 20.359],
    [26.668, 24.659],
    [18.909, 33.375],
    [18.25, 38.03],
    [22.507, 41.47],
    [27.501, 40.268],
    [33.248, 35.468],
    [35.851, 31.445],
    [38.519, 27.262],
    [36.684, 33.659],
    [39.352, 38.102],
    [42.133, 38.935],
    [48.07, 36.435],
    [55.229, 32.566],
    [60, 30]
  ],
  d: [
    [0, 30],
    [4.319, 30.792],
    [9.328, 31.626],
    [14.303, 32.459],
    [20.152, 32.459],
    [25.633, 30.792],
    [29.301, 28.291],
    [25.982, 28.946],
    [17.659, 32.003],
    [14.303, 38.667],
    [18.157, 41.365],
    [24.799, 39.5],
    [30.756, 33.67],
    [33.773, 28.698],
    [37.35, 21.528],
    [43.066, 9.541],
    [43.066, 6.605],
    [40.143, 10.374],
    [34.313, 24.131],
    [32.939, 32.837],
    [34.313, 37.833],
    [40.699, 39.933],
    [45.412, 38.003],
    [49.82, 34.909],
    [60, 30]
  ],
  f: [
    [0, 30],
    [14.221, 28.118],
    [28.072, 24.598],
    [45.87, 16.036],
    [50.036, 11.572],
    [48.369, 6.86],
    [43.133, 12.406],
    [36.672, 21.931],
    [33.715, 26.872],
    [28.072, 37.339],
    [25.148, 43.879],
    [25.475, 46.871],
    [28.549, 43.879],
    [32.882, 35.805],
    [36.214, 28.118],
    [37.849, 22.312],
    [36.644, 19.001],
    [39.206, 27.012],
    [42.538, 31.598],
    [50.036, 33.742],
    [60, 30]
  ],
  g: [
    [0, 30],
    [2.722, 29.935],
    [7.749, 30.769],
    [12.384, 31.602],
    [17.166, 32.2],
    [22.371, 30.16],
    [26.885, 27.714],
    [29.904, 25.729],
    [31.571, 20.333],
    [26.278, 22.422],
    [24.436, 26.965],
    [31.032, 24.895],
    [30.017, 28.548],
    [27.945, 34.718],
    [25.269, 41.182],
    [19.383, 51.106],
    [15.031, 55.483],
    [12.198, 54.65],
    [13.032, 51.106],
    [15.864, 47.085],
    [21.051, 41.589],
    [26.885, 36.385],
    [33.238, 30.994],
    [40.584, 26.965],
    [47.71, 24.895],
    [60, 30]
  ],
  h: [
    [0, 30],
    [4.488, 28.794],
    [17.83, 22.669],
    [25.213, 17.705],
    [26.975, 16.872],
    [31.645, 13.239],
    [36.112, 8.399],
    [32.804, 9.687],
    [24.38, 25.142],
    [22.324, 31.24],
    [21.49, 33.79],
    [19.498, 39.161],
    [18.203, 42.292],
    [16.556, 43.877],
    [17.83, 40.625],
    [23.546, 34.624],
    [26.842, 31.037],
    [30.478, 28.156],
    [34.277, 27.847],
    [34.668, 31.871],
    [34.082, 36.291],
    [35.111, 38.73],
    [40.306, 38.73],
    [48.238, 34.16],
    [52.8, 31.871],
    [60, 30]
  ],
  i: [
    [0, 30],
    [5.837, 30.997],
    [13.325, 31.871],
    [18.897, 32.56],
    [25.265, 32.887],
    [32.739, 32.887],
    [39.041, 32.053],
    [45.255, 30.171],
    [40.813, 37.125],
    [40.708, 41.827],
    [44.653, 42.223],
    [49.98, 39.63],
    [53.312, 35.827],
    [57.473, 32.053],
    [60, 30]
  ],
  l: [
    [0, 30],
    [11.466, 29.64],
    [19.357, 28.089],
    [26.806, 25.895],
    [32.622, 23.772],
    [38.332, 20.085],
    [42.27, 17.552],
    [45.954, 13.421],
    [47.621, 9.79],
    [46.788, 7.1],
    [44.561, 10.052],
    [37.97, 30],
    [35.577, 37.568],
    [34.744, 42.877],
    [39.316, 41.62],
    [44.561, 38.04],
    [50.113, 34.276],
    [55.692, 31.161],
    [60, 30]
  ],
  n: [
    [0, 30],
    [6.741, 30.879],
    [11.786, 31.713],
    [19.267, 32.546],
    [27.088, 32.198],
    [33.441, 30.683],
    [42.621, 26.405],
    [37.909, 32.546],
    [34.505, 37.528],
    [31.207, 40.051],
    [29.612, 40.012],
    [30.884, 37.002],
    [35.339, 32.198],
    [40.492, 27.894],
    [46.503, 25.572],
    [49.263, 27.06],
    [49.004, 35.861],
    [47.337, 38.362],
    [56.237, 31.517],
    [60, 30]
  ],
  q: [
    [0, 30],
    [8.163, 30.522],
    [18.557, 29.689],
    [26.017, 27.871],
    [35.027, 24.958],
    [41.089, 21.538],
    [41.849, 19.308],
    [39.936, 18.475],
    [32.685, 21.538],
    [29.349, 28.544],
    [31.208, 28.878],
    [36.867, 25.481],
    [40.024, 22.752],
    [34.194, 31.639],
    [29.187, 39.853],
    [27.277, 45.014],
    [27.936, 46.874],
    [30.375, 46.306],
    [34.917, 43.383],
    [40.182, 40.017],
    [45.345, 36.66],
    [49.388, 34.156],
    [60, 30]
  ],
  u: [
    [0, 30],
    [7.593, 27.55],
    [13.409, 25.841],
    [18.814, 22.511],
    [22.157, 20.046],
    [24.705, 18.553],
    [23.871, 21.678],
    [22.063, 25.361],
    [20.073, 30],
    [19.066, 33.381],
    [18.406, 38.302],
    [20.073, 41.752],
    [24.705, 40.919],
    [27.321, 39.136],
    [31.545, 35.048],
    [34.835, 31.401],
    [37.772, 27.365],
    [39.394, 25.583],
    [37.318, 32.235],
    [36.938, 36.045],
    [37.772, 39.495],
    [43.154, 39.969],
    [48.178, 37.145],
    [53.183, 34.215],
    [57.472, 31.401],
    [60, 30]
  ],
  y: [
    [0, 30],
    [5.637, 28.925],
    [10.209, 27.826],
    [12.591, 27.263],
    [14.974, 26.43],
    [19.124, 24.485],
    [21.738, 23.091],
    [22.951, 20.806],
    [20.223, 21.424],
    [12.28, 30],
    [14.719, 31.079],
    [18.924, 30.918],
    [24.423, 29.121],
    [31.434, 26.686],
    [37.75, 23.924],
    [42.474, 22.331],
    [39.531, 25.852],
    [37.252, 28.092],
    [32.73, 33.093],
    [29.679, 36.444],
    [26.926, 41.023],
    [22.179, 47.015],
    [11.758, 53.522],
    [6.47, 52.237],
    [19.957, 41.391],
    [35.03, 36.886],
    [52.337, 32.832],
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
