/** @type {import('next').NextConfig} */

const NodePolyfillPlugin = require("node-polyfill-webpack-plugin");
const CopyPlugin = require("copy-webpack-plugin");

const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  webpack: (config, {  }) => {

    config.resolve.extensions.push(".ts", ".tsx");
    config.resolve.fallback = { fs: false };

    config.plugins.push(
      new NodePolyfillPlugin(), 
      new CopyPlugin({
        patterns: [
          {
            from: './node_modules/onnxruntime-web/dist/ort-wasm.wasm',
            to: 'static/chunks/pages',
          },
          {
            from: './node_modules/onnxruntime-web/dist/ort-wasm-simd.wasm',
            to: 'static/chunks/pages',
          },
        ],
      }),
    );
    return config;
  },
  async redirects() {
    return [
      {
        source: '/ort-wasm.wasm',
        destination: '/_next/static/chunks/pages/ort-wasm.wasm',
        permanent: true,
      },
      {
        source: '/ort-wasm-simd.wasm',
        destination: '/_next/static/chunks/pages/ort-wasm-simd.wasm',
        permanent: true,
      },
    ]
  },
}

module.exports = nextConfig
