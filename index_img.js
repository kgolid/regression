const x_vals = [];
const y_vals = [];

const degree = 5;
const resolution = 100;

let learningRate, optimizer;
let theta;

let nx, ny;
let img, imgpixels;

const maxWidth = 800;
const maxHeight = 800;

let ready_for_processing;

let sketch = function (p) {
  p.setup = function () {
    const c = p.createCanvas(800, 800);
    ready_for_processing = false;

    p.noFill();

    learningRate = 0.2;
    optimizer = tf.train.adam(learningRate);
    theta = tf.randomUniform([degree]).variable();


    c.drop(gotFile, fileDropped);
    c.dragOver(fileDropped);
  };

  p.draw = function () {
    if (x_vals.length > 0) {
      optimizer.minimize(() => tf.tidy(() => loss(predict(x_vals), y_vals)));
    }

    p.background('#fec');
    draw_points(x_vals, y_vals);
    draw_prediction();
  };

  function draw_points(xs, ys) {
    p.stroke(255, 50, 50);
    p.strokeWeight(10);
    for (let i = 0; i < xs.length; i++) {
      const x = p.map(xs[i], -1, 1, 0, p.width);
      const y = p.map(ys[i], -1, 1, p.height, 0);
      p.point(x, y);
    }
  }

  function draw_line(xs, ys) {
    p.stroke(0);
    p.strokeWeight(3);
    p.beginShape();
    for (let i = 0; i < xs.length; i++) {
      const x = p.map(xs[i], -1, 1, 0, p.width);
      const y = p.map(ys[i], -1, 1, p.height, 0);
      p.vertex(x, y);
    }
    p.endShape();
  }

  function draw_prediction() {
    const x_range = [...Array(resolution + 1)].map((_, i) => p.map(i, 0, resolution, -1, 1));
    const predictions = tf.tidy(() => predict(x_range).dataSync());
    draw_line(x_range, predictions);
  }

  function gotFile(file) {
    if (file.type === 'image') {
      p.loadImage(file.data, imageLoaded);
    } else {
      console.log('Not an image file!');
    }
  }

  function imageLoaded(img) {
    if (img.height / maxHeight > img.width / maxWidth) {
      if (img.height > maxHeight) img.resize(0, maxHeight);
    } else {
      if (img.width > maxWidth) img.resize(maxWidth, 0);
    }

    img.loadPixels();

    nx = Math.floor(img.width);
    ny = Math.floor(img.height);
    p.resizeCanvas(nx, ny);

    imgpixels = newArray(ny).map((_, j) =>
      newArray(nx).map((_, i) => {
        var loc = (i + j * img.width) * 4;
        return [img.pixels[loc + 0], img.pixels[loc + 1], img.pixels[loc + 2]];
      })
    );
    console.log(imgpixels)
  }
};
new p5(sketch);

function predict(x_vals) {
  let xs = tf.tensor2d([x_vals]).transpose();
  let eyes = tf.ones([1, degree]);
  let range = tf.range(0, degree);
  
  return xs.matMul(eyes).pow(range).mul(theta).sum(1);
}

function loss(pred, y_vals) {
  return pred.sub(tf.tensor1d(y_vals)).square().mean();
}

function fileDropped() {
  ready_for_processing = false;
}

function newArray(n, value) {
  n = n || 0;
  var array = new Array(n);
  for (var i = 0; i < n; i++) {
    array[i] = value;
  }
  return array;
}
