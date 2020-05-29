const x_vals = [];
const y_vals = [];

const degree = 5;
const resolution = 100;

let learningRate, optimizer;
let theta;

let dragging;

let sketch = function (p) {
  p.setup = function () {
    p.createCanvas(800, 800);
    p.noFill();

    learningRate = 0.02;
    optimizer = tf.train.adam(learningRate);
    theta = tf.randomUniform([degree]).variable();

    dragging = false;
  };

  p.draw = function () {
    if (dragging) {
      const x = p.map(p.mouseX, 0, p.width, -1, 1);
      const y = p.map(p.mouseY, 0, p.height, 1, -1);

      x_vals.push(x);
      y_vals.push(y);
    } else if (x_vals.length > 0) {
      optimizer.minimize(() => tf.tidy(() => loss(predict(x_vals), y_vals)));
    }

    p.background('#fec');
    draw_points(x_vals, y_vals);
    draw_prediction();
  };

  p.mousePressed = () => dragging = true;

  p.mouseReleased = () => dragging = false;

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
