let counter = 0;

class MyImg {
  constructor(type, index) {
    let img_scale = 0.2;
    // let index = 0;
    switch (type) {
      case 'A':
        // index = Math.floor(random(1, 4));
        this.img_1 = loadImage('assets/A/' + index + '-1.png');
        this.img_2 = loadImage('assets/A/' + index + '-2.png', _ => {
          this.x = 0;
          this.y = this.img_2.height * img_scale - 100;

          this.width = this.img_2.width * img_scale;
          this.height = this.img_2.height * img_scale;
        });
        break;
      case 'B':
        // index = Math.floor(random(1, 7));
        this.img_1 = loadImage('assets/B/' + index + '-1.png');
        this.img_2 = loadImage('assets/B/' + index + '-2.png', _ => {
          this.x = 0 - this.img_2.width * img_scale + 80;
          this.y = 260;

          this.width = this.img_2.width * img_scale;
          this.height = this.img_2.height * img_scale;
        });
        break;
      case 'C':
        // index = Math.floor(random(1, 7));
        this.img_1 = loadImage('assets/C/' + index + '-1.png');
        this.img_2 = loadImage('assets/C/' + index + '-2.png', _ => {
          this.x = 0 - this.img_2.width * img_scale + 50;
          this.y = 480;

          this.width = this.img_2.width * img_scale;
          this.height = this.img_2.height * img_scale;
        });
        break;
      case 'D':
        // ok, we do special cases for D
        // index = Math.floor(random(1, 4));
        this.img_1 = loadImage('assets/D/' + index + '-1.png');
        this.img_2 = loadImage('assets/D/' + index + '-2.png', _ => {
          this.x = 0;
          switch (index) {
            case 1:
              this.y = 650;
              break;
            case 2:
              this.y = 800;
              break;
            case 3:
              this.y = 700;
              break;
          }

          this.width = this.img_2.width * img_scale;
          this.height = this.img_2.height * img_scale;
        });

        break;
      case 'E':
        // index = Math.floor(random(1, 4));
        this.img_1 = loadImage('assets/E/' + index + '-1.png');
        this.img_2 = loadImage('assets/E/' + index + '-2.png', _ => {
          this.x = 0;
          this.y = 380;

          this.width = this.img_2.width * img_scale;
          this.height = this.img_2.height * img_scale;
        });

        break;
    }
  }

  randomColor() {
    let colors = [Math.floor(random(50, 255)), Math.floor(random(50, 255)), 255];
    colors.sort(() => Math.random() - 0.5);

    this.colors = colors;
  }

  display() {
    this.randomColor();
    tint(this.colors[0], this.colors[1], this.colors[2], 255 * 0.66);
    image(this.img_1, this.x, this.y, this.width, this.height);
    noTint()
    image(this.img_2, this.x, this.y, this.width, this.height);
  }
}

function preload() {
  // Load the image
  img_as = [new MyImg('A', 1), new MyImg('A', 2), new MyImg('A', 3)];
  img_bs = [new MyImg('B', 1), new MyImg('B', 2), new MyImg('B', 3), new MyImg('B', 4), new MyImg('B', 5), new MyImg('B', 6)];
  img_cs = [new MyImg('C', 1), new MyImg('C', 2), new MyImg('C', 3), new MyImg('C', 4), new MyImg('C', 5), new MyImg('C', 6)];
  img_ds = [new MyImg('D', 1), new MyImg('D', 2), new MyImg('D', 3)];
  img_es = [new MyImg('E', 1), new MyImg('E', 2), new MyImg('E', 3)];
}

function setup() {
  createCanvas(1400, 1400);
  background(0, 0);

  imageMode(CENTER);
  noLoop();
}

function draw() {
  translate(width / 2, 0)
  img_a = img_as[Math.floor(random(0, 3))];
  img_b = img_bs[Math.floor(random(0, 6))];
  img_c = img_cs[Math.floor(random(0, 6))];
  img_d = img_ds[Math.floor(random(0, 3))];
  img_e = img_es[Math.floor(random(0, 3))];

  // img_a = img_as[1];
  // img_b = img_bs[1];
  // img_c = img_cs[1];
  // img_d = img_ds[1];
  // img_e = img_es[1];

  // display images
  img_b.display();
  img_c.display();

  // mirror b and c
  push();
  scale(-1, 1);
  img_b.display();
  img_c.display();
  pop();

  img_a.display();
  img_d.display();
  img_e.display();
}

function mousePressed() {
  // saveCanvas('sketch' + counter, "png");
  clear();
  redraw();
}

function keyPressed() {
  if (key == 's') {
    counter +=1 ;
    saveCanvas('sketch' + counter, "png");
  }
}