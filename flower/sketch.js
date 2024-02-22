class MyImg {
  constructor(type) {
    let img_scale = 0.1;
    switch (type) {
      case 'A':
        this.img = loadImage('assets/A/' + Math.floor(random(1, 4)) + '.png', _ => {
          this.x = 0;
          this.y = this.img.height * img_scale;
        });
        break;
      case 'B':
        this.img = loadImage('assets/B/' + Math.floor(random(1, 7)) + '.png', _ => {
          this.x = 0 - this.img.width * img_scale + 50;
          this.y = 200;
        });
        break;
      case 'C':
        this.img = loadImage('assets/C/' + Math.floor(random(1, 7)) + '.png', _ => {
          this.x = 0 - this.img.width * img_scale + 50;
          this.y = 320;
        });
        break;
      case 'D':
        // ok, we do special cases for D
        let index = Math.floor(random(1, 4));
        this.img = loadImage('assets/D/' + index + '.png', _ => {
          this.x = 0;
          switch (index) {
            case 1:
              this.y = 480;
              break;
            case 2:
              this.y = 425;
              break;
            case 3:
              this.y = 400;
              break;
          }
        });

        break;
      case 'E':
        this.img = loadImage('assets/E/' + Math.floor(random(1, 4)) + '.png', _ => {
          this.x = 0;
          this.y = 270;
        });

        break;
    }

    let colors = [Math.floor(random(50, 255)), Math.floor(random(50, 255)), 255];
    colors.sort(() => Math.random() - 0.5);
    this.colors = colors;
  }

  display() {
    let img_scale = 0.1;
    // let img = setImgColor(this.img, color(this.colors[0], this.colors[1], this.colors[2], 200));
    // image(img, this.x, this.y, this.img.width * img_scale, this.img.height * img_scale);

    tint(this.colors[0], this.colors[1], this.colors[2], 200);
    image(this.img, this.x, this.y, this.img.width * img_scale, this.img.height * img_scale);
  }
}

function preload() {
  // Load the image

  img_a = new MyImg('A');
  img_b = new MyImg('B');
  img_c = new MyImg('C');
  img_d = new MyImg('D');
  img_e = new MyImg('E');
}

function setup() {
  createCanvas(800, 1000);
  imageMode(CENTER);
  noLoop();
}

function draw() {
  background('white');
  translate(width / 2, 0)

  // display images
  img_e.display();
  img_d.display();

  img_b.display();
  img_c.display();

  // mirror b and c
  push();
  scale(-1, 1);
  img_b.display();
  img_c.display();
  pop();

  img_a.display();
}

function setImgColor(img, replaceColor) {
  img.loadPixels();
  for (y = 0; y < img.height; y++) {
    for (x = 0; x < img.width; x++) {
      index = (x + y * img.width) * 4;
      // 获取当前像素的颜色
      currentColor = color(img.pixels[index], img.pixels[index + 1], img.pixels[index + 2], img.pixels[index + 3]);
      if (alpha(currentColor) > 0 && brightness(currentColor) === 0) {
        img.set(x, y, replaceColor);
      }
    }
  }
  img.updatePixels();
  return img;
}