# PyTermRenderer
## _Terminal Renderer Written in Python3_

PyTermRenderer, written in Python3, is a tool to render images, or to create and render some awesome effects in 256-color terminals.

## Modules & APIs

- backend.py - providing some implemented ascii effects APIs
  - `TheMatrixCodeRain` - a class providing ascii letters rain effect like the one in *The Matrix*
  - `DigitalTimeUnit` - a class providing digital clock effect
- controller.py - providing APIs to capture or to handle key input events
  - `KeyboardListener` - a class providing methods to capture key input events, or to bind callbacks to key input events
- decoder.py - providing APIs to decode PNG images
  - `PNG` - a decoder class for retriving data from single PNG file
  - `PNGSequence` - a decoder class for retriving data from multiple PNG files, A.K.A PNG sequence
- hintings.py - custom type bindings for IDE level type hinting
- main.py - the entry point of this project
- renderer.py - providing different kinds of rendering APIs
  - `clear_terminal` - a function to clear terminal
  - `render_frame` - a function to render a frame of RGBA colors and characters
  - `render_ascii` - a function to render a frame of RGBA colors with ASCII letters
  - `render_gray` - a function to  render a frame of RGBA colors in grayscale mode

## Features

- Performance counted on
- No third-party lib needed
- Well type hinted, thus IDE friendly

## References

1. Implementation of the `PNG` decoder had referenced [Python-3D-renderer](https://github.com/ICE27182/Python-3D-renderer).