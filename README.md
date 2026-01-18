# CliDrop 

**Real-time audio spectrum visualizer for the terminal with multiple visualization modes, particle effects, and psychedelic color cycling.**

![Python](https://img.shields.io/badge/python-3.6+-green) ![License](https://img.shields.io/badge/license-MIT-orange)

## What it is...

It's an Audio Visualizer for use in a terminal emulator or TTY. It catches the sound of your system and draws nice graphics, just for fun.

Adjust your terminal size, select a nice character and have fun listening to your favorite music. Combine clidrop, with tmux, to create even nicer visual fxs, with horizontal/vertical panes and running multiple instances of clidrop.

It can run in any terminal, even bare core TTY, so it's the best choice to use wiht SBC like the RPi0.

Demo: [Youtube Video](https://www.youtube.com/watch?v=4xLkTYtUw6U)

## Features

- **10 unique visualization modes** - from classic frequency bars to particle systems
- **Real-time audio processing** - reacts instantly to your system audio
- **Stereo & mono support** - visualizes both channels separately
- **Interactive controls** - change modes, colors, and effects on the fly
- **Customizable display** - choose your character, colors, and particle count
- **Psychedelic mode** - automatic color cycling for trippy effects
- **Reverse mode** - flip visualization direction
- **Command-line friendly** - launch with specific settings

## Quick Start

### Installation

It requires minimal dependencies, so it's able to run on any system that has Python.

```bash
# Clone the repository
git clone https://github.com/yourusername/clidrop.git
cd clidrop

# Install dependencies
pip install numpy sounddevice

# Run CliDrop
python clidrop.py
```

### Basic Usage

```bash

# Start with default settings (mode 1: normal bars)
python clidrop.py

# Start in particle mode with custom settings
python clidrop.py -m 7 --particles 100 -c "@" -fg 2

# Enable psychedelic and reverse modes
python clidrop.py -p -r
```

### Visualization Modes

```
Mode	Name	            Description
  0	  EQ Curve	        Smooth frequency response curves
  1	  Normal Bars	      Classic frequency bar display
  2	  Centered Bars	    Bars centered around middle line
  3	  Circle	          Single circle visualization
  4	  Waveform	        Audio waveform points
  5	  Bar Waveform	    Full bar waveform
  6	  Dual Circles	    Stereo circle visualization
  7	  Particles	        Animated particle system
  8	  Binary	          On/off binary display
  9	  Lissajous	        X-Y oscilloscope patterns
```

### Keys

While CliDrop is running:
```
Key	    Action
0-9	    Switch visualization mode
f	      Cycle foreground colors
b	      Cycle background colors
p	      Toggle psychedelic mode
r	      Toggle reverse mode
q	      Quit CliDrop
```

### Command Line Options

```bash

python clidrop.py [OPTIONS]

Options:
  -m, --mode MODE        Drawing mode from 0 to 9 (default: 1)
  -r, --reverse          Enable reverse mode
  -p, --psychedelic      Enable psychedelic mode
  -c, --char CHAR        Character to display for bars (default: ▌)
  -fg, --foreground FG   Foreground color (0-7, default: 7)
  -bg, --background BG   Background color (0-7, default: 0)
  --particles COUNT      Number of particles (mode 7, default: 50)
  -h, --help            Show help message
```

### Examples

Use the command line to assing different drawing character to each mode. Some modes, look nicer with specific characters than others.

Enable the Pshychedelic mode, for even cooler visuals!

```bash

# Professional EQ curve display
python clidrop.py -m 0 -c "█" -fg 6 -bg 0

# Psychedelic particle system
python clidrop.py -m 7 -p --particles 200

# Retro terminal vibe
python clidrop.py -m 1 -c "#" -fg 2 -bg 0

# Stereo visualization
python clidrop.py -m 6 -c "○"

# Binary mode for minimalist look
python clidrop.py -m 8 -c "█" -fg 1
```

### Dependencies

- Python 3.6+
- numpy - Numerical computations
- sounddevice - Audio input capture

Install all dependencies:
```bash

pip install -r requirements.txt
```
