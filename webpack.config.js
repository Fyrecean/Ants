const path = require('path');

module.exports = {
    mode: "development",
    entry: './src/main.ts',
    devtool: 'inline-source-map',
    module: {
        rules: [
            {
                test: /\.wgsl$/,         // Apply raw-loader to .wgsl files
                use: 'raw-loader',
              },
              {
                test: /\.ts$/,           // Apply ts-loader to .ts files
                use: 'ts-loader',
                exclude: /node_modules/, // Exclude node_modules from ts-loader
              },
        ],
    },
    resolve: {
        extensions: [".ts"]
    },
    output: {
        filename: 'bundle.js',
        path: path.resolve(__dirname, "public"),
    }
};