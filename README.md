# Neural Spectral Automata

This repo contains the source code for [Neural Spectral Automata](https://chrisades.github.io/Neural-Spectral-Automata), A web tool for creating 1D cellular automata and turning them into sound. Each row of the automaton is generated using a simple neural-style rule (convolution → activation → constraint), and the resulting image is read as an evolving sound spectrum, turning chaotic visual patterns into audio.

## Features

- Adjustable grid size (width/height) controlling pattern detail and audio length
- Multiple initialization modes: random, random-select, and impulse
- Customizable convolution mask weights
- Several activation functions (linear, sin, tanh, power, and more) with tunable parameters
- Wrap-around and clip/modulo boundary controls
- Audio generation via inverse FFT, with adjustable time stretch
- Save/load custom presets
- Download generated images and audio

## Credits

Created by [Chris Ades](https://chrisades.github.io). Inspired by Emergent Garden's [neuralpatterns.io](https://neuralpatterns.io/).
