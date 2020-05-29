import pkg from './package.json';
import resolve from 'rollup-plugin-node-resolve';
import commonjs from 'rollup-plugin-commonjs';

export default [
  {
    input: 'index.js',
    output: {
      file: pkg.browser,
      format: 'umd'
    },
    plugins: [resolve(), commonjs()]
  },
  {
    input: 'index_img.js',
    output: {
      file: pkg.browser_img,
      format: 'umd'
    },
    plugins: [resolve(), commonjs()]
  }
];