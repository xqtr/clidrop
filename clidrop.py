#!/usr/bin/env python3
import numpy as np
import sounddevice as sd
import shutil
import sys
import time
import tty
import termios
import threading
import math
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Audio Visualizer for Terminal')
parser.add_argument('-m', '--mode', type=int, default=1, choices=range(10), 
                    help='Drawing mode from 0 to 9 (default: 1)')
parser.add_argument('-r', '--reverse', action='store_true',
                    help='Enable reverse mode')
parser.add_argument('-p', '--psychedelic', action='store_true',
                    help='Enable psychedelic mode')
parser.add_argument('-c', '--char', type=str, default='█',
                    help='Character to display for bars (default: ▌)')
parser.add_argument('-fg', '--foreground', type=int, default=7, choices=range(8),
                    help='Foreground color (0 to 7, default: 7)')
parser.add_argument('-bg', '--background', type=int, default=0, choices=range(8),
                    help='Background color (0 to 7, default: 0)')
parser.add_argument('--particles', type=int, default=50,
                    help='Number of particles to initialize (default: 50)')

args = parser.parse_args()

RATE = 44100
BLOCK = 1024

cols, rows = shutil.get_terminal_size()
BARS = cols #// 2
MAX_H = rows - 2

# Use command line character argument
CHAR = args.char

mode = args.mode
running = True
reverse = args.reverse
psychedelic = args.psychedelic
psychdelay = 5
psychcount = 0

particles = []
PARTICLE_COUNT = args.particles
particle_colors = [31, 32, 33, 34, 35, 36, 91, 92, 93, 94, 95, 96]  # Red, Green, Yellow, Blue, Magenta, Cyan + bright versions
PARTICLE_COLOR = (32,33,31)

fg_colors = [30,31,32,33,34,35,36,37]
bg_colors = [40,41,42,43,44,45,46,47]

# Use command line color arguments
fg_i = args.foreground
bg_i = args.background

def apply_colors():
    sys.stdout.write(f"\x1b[{fg_colors[fg_i]};{bg_colors[bg_i]}m")

def key_listener():
    global mode, running, fg_i, bg_i, psychedelic, reverse, PARTICLE_COUNT, PARTICLE_COLOR
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while running:
            c = sys.stdin.read(1)
            if c in "1234567890": mode = int(c)
            elif c == "f":
                fg_i = (fg_i + 1) % len(fg_colors)
                if fg_i == 0:
                    PARTICLE_COLOR = (32,33,31)
                elif fg_i == 1:
                    PARTICLE_COLOR = (31,35,34)
                elif fg_i == 3:
                    PARTICLE_COLOR = (30,33,31)
                elif fg_i == 4:
                    PARTICLE_COLOR = (36,34,30)
                elif fg_i == 5:
                    PARTICLE_COLOR = (33,33,36)
                elif fg_i == 6:
                    PARTICLE_COLOR = (33,31,35)
                elif fg_i == 7:
                    PARTICLE_COLOR = (34,36,33)
                    
            elif c == "b": bg_i = (bg_i + 1) % len(bg_colors)
            elif c == "q": running = False
            elif c == "p": psychedelic = not psychedelic
            elif c == "r": reverse = not reverse
            
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def clear():
    sys.stdout.write("\x1b[H")
    
def init_particles():
    global particles
    particles = []
    for _ in range(PARTICLE_COUNT):
        particles.append({
            'x': np.random.uniform(0, cols-1),
            'y': np.random.uniform(0, rows-1),
            'vx': np.random.uniform(-0.5, 0.5),
            'vy': np.random.uniform(-0.5, 0.5),
            'life': np.random.uniform(0.5, 1.0),
            'color': np.random.choice(particle_colors),
            'size': np.random.choice(['.', 'o', 'O', '@']),
            'frequency_band': np.random.randint(0, 8),  # 0-7 frequency bands
        })

# Initialize particles at start
init_particles()

def draw_particles_basic(vals_left, vals_right=None):
    clear()
    global particles, PARTICLE_COLOR
    
    # Get overall energy from binned values (not raw FFT)
    if vals_right is not None:
        # Average left and right channels
        avg_vals = [(l + r) / 2 for l, r in zip(vals_left[:len(vals_right)], vals_right[:len(vals_left)])]
        if avg_vals:
            energy = sum(avg_vals) / len(avg_vals)
        else:
            energy = 0
    else:
        if vals_left:
            energy = sum(vals_left) / len(vals_left)
        else:
            energy = 0
    
    # Normalize energy - adjust scaling factor based on your typical values
    # Typical vals_left values are 0-10 after log scaling, so divide by 5-10
    energy_norm = min(1.0, energy / 10.0)  # Adjust 5.0 as needed
    
    # If there's no sound, use minimal energy
    if energy_norm < 0.01:
        energy_norm = 0.01
    
    # Initialize particles if empty
    if not particles:
        init_particles()
    
    # Update particles
    canvas = [[" "]*cols for _ in range(rows)]
    
    for particle in particles:
        # Random movement influenced by audio energy
        particle['vx'] += np.random.uniform(-0.3, 0.3) * (0.5 + energy_norm)
        particle['vy'] += np.random.uniform(-0.3, 0.3) * (0.5 + energy_norm)
        
        # Gravity-like force toward center when energy is high
        if energy_norm > 0.3:
            dx = cols/2 - particle['x']
            dy = rows/2 - particle['y']
            dist = max(0.1, math.sqrt(dx*dx + dy*dy))
            particle['vx'] += dx / dist * 0.1 * energy_norm
            particle['vy'] += dy / dist * 0.1 * energy_norm
        
        # Damping
        particle['vx'] *= 0.92
        particle['vy'] *= 0.92
        
        # Update position
        particle['x'] += particle['vx']
        particle['y'] += particle['vy']
        
        # Bounce at edges
        if particle['x'] < 0:
            particle['x'] = 0
            particle['vx'] = abs(particle['vx']) * 0.8
        elif particle['x'] >= cols:
            particle['x'] = cols - 1
            particle['vx'] = -abs(particle['vx']) * 0.8
        
        if particle['y'] < 0:
            particle['y'] = 0
            particle['vy'] = abs(particle['vy']) * 0.8
        elif particle['y'] >= rows:
            particle['y'] = rows - 1
            particle['vy'] = -abs(particle['vy']) * 0.8
        
        # Calculate distance from center for color
        dx = particle['x'] - cols/2
        dy = particle['y'] - rows/2
        dist_from_center = math.sqrt(dx*dx + dy*dy) / (min(cols, rows)/2)
        
        # Choose color based on position and energy
        if dist_from_center < 0.3:
            color = PARTICLE_COLOR[0] #32  # Green in center
        elif dist_from_center < 0.6:
            color = PARTICLE_COLOR[1] #33  # Yellow in middle
        else:
            color = PARTICLE_COLOR[2] #31  # Red at edges
        
        # Make color brighter with higher energy
        if energy_norm > 0.8:
            color += 60  # Bright colors

        # Choose character based on energy
        if energy_norm > 0.8:
            char = '@'
        elif energy_norm > 0.6:
            char = '8'
        elif energy_norm > 0.4:
            char = 'O'
        elif energy_norm > 0.2:
            char = 'o'
        else:
            char = '.'
        
        # Draw particle
        x = int(particle['x'])
        y = int(particle['y'])
        
        if 0 <= x < cols and 0 <= y < rows:
            # Main particle
            canvas[y][x] = f"\x1b[{color}m{char}\x1b[0m"
            
            # Trail effect (commented out for simplicity)
            # for i in range(1, 3):
            #     trail_x = max(0, min(cols-1, x - int(particle['vx'] * i)))
            #     trail_y = max(0, min(rows-1, y - int(particle['vy'] * i)))
            #     if canvas[trail_y][trail_x] == " ":
            #         trail_char = ['.', ':'][min(i-1, 1)]
            #         canvas[trail_y][trail_x] = f"\x1b[{color-10}m{trail_char}\x1b[0m"
    
    # Draw canvas
    for y in range(rows):
        for x in range(cols):
            if canvas[y][x] != " ":
                sys.stdout.write(canvas[y][x])
            else:
                # Add subtle background pattern
                if (x + y) % 8 == 0:
                    #sys.stdout.write("\x1b[90m·\x1b[0m")  # Dark gray dots
                    sys.stdout.write("\x1b[90m \x1b[0m")  # Dark gray dots
                else:
                    sys.stdout.write(" ")
        sys.stdout.write("\n")
    sys.stdout.flush()

def draw_normal(vals_left, vals_right=None):
    clear(); apply_colors()
    
    if vals_right is not None:
        # Stereo mode - left and right side by side
        for h in range(MAX_H, 0, -1):
            # Left channel (first half)
            for v in vals_left[:len(vals_left)//2]:
                sys.stdout.write(CHAR if v >= h else " ")
            # Right channel (second half)
            for v in vals_right[len(vals_right)//2:]:
                sys.stdout.write(CHAR if v >= h else " ")
            sys.stdout.write("\n")
    else:
        # Mono mode
        for h in range(MAX_H, 0, -1):
            for v in vals_left:
                sys.stdout.write(CHAR if v >= h else " ")
            sys.stdout.write("\n")
    sys.stdout.flush()
    
def draw_normal_reverse(vals_left, vals_right=None):
    clear(); apply_colors()
    
    if vals_right is not None:
        # Stereo mode - left and right side by side, reversed
        for h in range(MAX_H, 0, -1):
            # Right channel (second half) - drawn on LEFT side when reversed
            for v in reversed(vals_right[len(vals_right)//2:]):
                sys.stdout.write(CHAR if v >= h else " ")
            # Left channel (first half) - drawn on RIGHT side when reversed
            for v in reversed(vals_left[:len(vals_left)//2]):
                sys.stdout.write(CHAR if v >= h else " ")
            sys.stdout.write("\n")
    else:
        # Mono mode - reversed
        for h in range(MAX_H, 0, -1):
            for v in reversed(vals_left):
                sys.stdout.write(CHAR if v >= h else " ")
            sys.stdout.write("\n")
    sys.stdout.flush()
    
def draw_centered(vals_left, vals_right=None):
    clear(); apply_colors()
    canvas = [[" "]*cols for _ in range(rows)]
    mid = rows // 2
    
    
    if vals_right is not None:
        # Stereo mode
        left_cols = cols // 2
        right_cols = cols // 2
        
        # Left channel (first half)
        for i, v in enumerate(vals_left[:left_cols]):
            x = i
            for h in range(v):
                if 0 <= mid-h < rows: canvas[mid-h][x] = CHAR
                if 0 <= mid+h < rows: canvas[mid+h][x] = CHAR
        
        # Right channel (second half)
        for i, v in enumerate(vals_right[-right_cols:]):
            x = left_cols + i
            for h in range(v):
                if 0 <= mid-h < rows: canvas[mid-h][x] = CHAR
                if 0 <= mid+h < rows: canvas[mid+h][x] = CHAR
    else:
        # Mono mode
        for i, v in enumerate(vals_left):
            x = i
            for h in range(v):
                if 0 <= mid-h < rows: canvas[mid-h][x] = CHAR
                if 0 <= mid+h < rows: canvas[mid+h][x] = CHAR
    
    for r in canvas: sys.stdout.write("".join(r) + "\n")
    sys.stdout.flush()
    
def draw_centered_reverse(vals_left, vals_right=None):
    clear(); apply_colors()
    canvas = [[" "]*cols for _ in range(rows)]
    mid = rows // 2
    
    if vals_right is not None:
        # Stereo mode - reversed with channel swapping
        left_cols = cols // 2
        right_cols = cols // 2
        
        # Right channel (second half) - drawn on LEFT side when reversed
        # Note: We need to reverse the order of values AND swap channels
        right_vals_reversed = list(reversed(vals_right[-right_cols:]))
        for i, v in enumerate(right_vals_reversed):
            x = i  # Right channel goes to left side
            for h in range(v):
                if 0 <= mid-h < rows: canvas[mid-h][x] = CHAR
                if 0 <= mid+h < rows: canvas[mid+h][x] = CHAR
        
        # Left channel (first half) - drawn on RIGHT side when reversed
        left_vals_reversed = list(reversed(vals_left[:left_cols]))
        for i, v in enumerate(left_vals_reversed):
            x = left_cols + i  # Left channel goes to right side
            for h in range(v):
                if 0 <= mid-h < rows: canvas[mid-h][x] = CHAR
                if 0 <= mid+h < rows: canvas[mid+h][x] = CHAR
    else:
        # Mono mode - just reversed
        for i, v in enumerate(reversed(vals_left)):
            x = i
            for h in range(v):
                if 0 <= mid-h < rows: canvas[mid-h][x] = CHAR
                if 0 <= mid+h < rows: canvas[mid+h][x] = CHAR
    
    for r in canvas: sys.stdout.write("".join(r) + "\n")
    sys.stdout.flush()

def draw_circle(radius):
    clear(); apply_colors()
    cx, cy = cols//2, rows//2
    canvas = [[" "]*cols for _ in range(rows)]
    for y in range(rows):
        for x in range(cols):
            d = math.hypot(x-cx, (y-cy)*2)
            if abs(d-radius) < 1.0:
                canvas[y][x] = CHAR
    for r in canvas: sys.stdout.write("".join(r) + "\n")
    sys.stdout.flush()

def draw_circle_dual(radius_left, radius_right=None):
    clear(); apply_colors()
    canvas = [[" "]*cols for _ in range(rows)]
    
    if radius_right is not None:
        # Stereo mode - two circles side by side
        left_cx = cols // 4
        right_cx = 3 * cols // 4
        cy = rows // 2
        
        for y in range(rows):
            for x in range(cols):
                # Left circle
                d_left = math.hypot(x-left_cx, (y-cy)*2)
                if abs(d_left-radius_left) < 1.0:
                    canvas[y][x] = CHAR
                
                # Right circle  
                d_right = math.hypot(x-right_cx, (y-cy)*2)
                if abs(d_right-radius_right) < 1.0:
                    canvas[y][x] = CHAR
    else:
        # Mono mode - single centered circle
        cx, cy = cols//2, rows//2
        for y in range(rows):
            for x in range(cols):
                d = math.hypot(x-cx, (y-cy)*2)
                if abs(d-radius_left) < 1.0:
                    canvas[y][x] = CHAR
    
    for r in canvas: sys.stdout.write("".join(r) + "\n")
    sys.stdout.flush()

def draw_waveform(left_data, right_data=None):
    clear(); apply_colors()
    canvas = [[" "]*cols for _ in range(rows)]
    mid = rows // 2
    
    if right_data is not None:
        # Stereo mode
        left_cols = cols // 2
        right_cols = cols // 2
        
        # Left channel (first half)
        step_left = max(1, len(left_data)//left_cols)
        for x in range(left_cols):
            y = int(left_data[x*step_left] * (rows//2))
            if 0 <= mid-y < rows:
                canvas[mid-y][x] = CHAR
        
        # Right channel (second half)
        step_right = max(1, len(right_data)//right_cols)
        for x in range(right_cols):
            y = int(right_data[x*step_right] * (rows//2))
            col_pos = left_cols + x
            if 0 <= mid-y < rows:
                canvas[mid-y][col_pos] = CHAR
    else:
        # Mono mode
        step = max(1, len(left_data)//cols)
        for x in range(cols):
            y = int(left_data[x*step] * (rows//2))
            if 0 <= mid-y < rows:
                canvas[mid-y][x] = CHAR
    
    for r in canvas: sys.stdout.write("".join(r) + "\n")
    sys.stdout.flush()

def draw_bar_waveform(left_data, right_data=None):
    clear(); apply_colors()
    canvas = [[" "]*cols for _ in range(rows)]
    mid = rows // 2
    
    if right_data is not None:
        # Stereo - draw side-by-side bars
        left_cols = cols // 2
        right_cols = cols // 2
        
        # Left channel (first half)
        step_left = max(1, len(left_data)//left_cols)
        for x in range(left_cols):
            sample = left_data[x*step_left]
            height = int(abs(sample) * (rows//2))
            if height > 0:
                for y in range(mid-height, mid+height+1):
                    if 0 <= y < rows:
                        canvas[y][x] = CHAR
        
        # Right channel (second half)
        step_right = max(1, len(right_data)//right_cols)
        for x in range(right_cols):
            sample = right_data[x*step_right]
            height = int(abs(sample) * (rows//2))
            if height > 0:
                col_pos = left_cols + x
                for y in range(mid-height, mid+height+1):
                    if 0 <= y < rows:
                        canvas[y][col_pos] = CHAR
    else:
        # Mono - draw single bars
        step = max(1, len(left_data)//cols)
        for x in range(cols):
            sample = left_data[x*step]
            height = int(abs(sample) * (rows//2))
            if height > 0:
                for y in range(mid-height, mid+height+1):
                    if 0 <= y < rows:
                        canvas[y][x] = CHAR
    
    for r in canvas: sys.stdout.write("".join(r) + "\n")
    sys.stdout.flush()
    
def draw_lissajous(left_data, right_data=None):
    clear(); apply_colors()
    canvas = [[" "]*cols for _ in range(rows)]
    
    if right_data is not None:
        # Stereo mode: Plot left channel vs right channel (X-Y mode)
        # Scale factors to fit within screen
        x_scale = cols / 2
        y_scale = rows / 2
        x_offset = cols / 2
        y_offset = rows / 2
        
        # Use a subset of samples for better performance
        step = max(1, len(left_data) // (cols * 2))
        
        for i in range(0, min(len(left_data), len(right_data)), step):
            # Convert audio samples to screen coordinates
            # x position based on left channel
            x = int(left_data[i] * x_scale + x_offset)
            # y position based on right channel
            y = int(right_data[i] * y_scale + y_offset)
            
            # Clamp to screen bounds
            x = max(0, min(cols - 1, x))
            y = max(0, min(rows - 1, y))
            
            # Draw point
            canvas[y][x] = CHAR
        
        # For mono input, we can also draw a simpler version
    else:
        # Mono mode: Simulate Lissajous with phase shift
        # Create artificial phase-shifted version of the signal
        x_scale = cols / 2
        y_scale = rows / 2
        x_offset = cols / 2
        y_offset = rows / 2
        
        step = max(1, len(left_data) // (cols * 2))
        phase_shift = len(left_data) // 4  # 90 degree phase shift
        
        for i in range(0, len(left_data) - phase_shift, step):
            # Original signal for X
            x = int(left_data[i] * x_scale + x_offset)
            # Phase-shifted signal for Y
            y = int(left_data[i + phase_shift] * y_scale + y_offset)
            
            # Clamp to screen bounds
            x = max(0, min(cols - 1, x))
            y = max(0, min(rows - 1, y))
            
            # Draw point
            canvas[y][x] = CHAR
    
    for r in canvas: sys.stdout.write("".join(r) + "\n")
    sys.stdout.flush()
    
def draw_eq_curve(vals_left, vals_right=None):
    clear(); apply_colors()
    
    # Interpolate between points to create a smooth curve
    def interpolate_points(points, num_points):
        if len(points) < 2:
            return points
        
        x_coords, y_coords = zip(*points)
        # Create smooth curve using simple interpolation
        smooth_x = np.linspace(min(x_coords), max(x_coords), num_points)
        smooth_y = np.interp(smooth_x, x_coords, y_coords)
        
        return list(zip(smooth_x, smooth_y))
    
    # Bresenham's line algorithm for drawing lines between points
    def draw_line(x0, y0, x1, y1, canvas, char):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            if 0 <= x0 < cols and 0 <= y0 < rows:
                canvas[y0][x0] = char
            
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
    
    canvas = [[" "]*cols for _ in range(rows)]
    #mid = rows // 2
    mid = rows -5
    
    # Helper function to get safe max value
    def safe_max(values, default=1):
        if not values:
            return default
        max_val = max(values)
        return max_val if max_val > 0 else default
    
    if vals_right is not None:
        # Stereo mode - two curves side by side
        left_cols = cols // 2
        right_cols = cols // 2
        
        # Get left and right values (ensure they exist)
        left_vals = vals_left[:left_cols] if len(vals_left) >= left_cols else vals_left
        right_vals = vals_right[-right_cols:] if len(vals_right) >= right_cols else vals_right
        
        # Use safe max values to avoid division by zero
        max_left_val = safe_max(left_vals, 1)
        max_right_val = safe_max(right_vals, 1)
        
        # Left channel curve points
        left_points = []
        for i in range(min(len(left_vals), left_cols)):
            v = left_vals[i]
            # Scale X to left half, Y to vertical range
            x = int((i / max(1, len(left_vals)-1)) * (left_cols - 1))
            y = int((v / max_left_val) * (rows // 2))
            # Center vertically and clamp
            y_pos = mid - y
            y_pos = max(0, min(rows-1, y_pos))
            left_points.append((x, y_pos))
        
        # Right channel curve points
        right_points = []
        for i in range(min(len(right_vals), right_cols)):
            v = right_vals[i]
            # Scale X to right half
            x = left_cols + int((i / max(1, len(right_vals)-1)) * (right_cols - 1))
            y = int((v / max_right_val) * (rows // 2))
            y_pos = mid - y
            y_pos = max(0, min(rows-1, y_pos))
            right_points.append((x, y_pos))
        
        # Only create curves if we have points
        if left_points:
            # Create smooth curves
            smooth_left = interpolate_points(left_points, left_cols)
            
            # Draw left curve
            for i in range(len(smooth_left) - 1):
                x0, y0 = int(smooth_left[i][0]), int(smooth_left[i][1])
                x1, y1 = int(smooth_left[i+1][0]), int(smooth_left[i+1][1])
                #draw_line(x0, y0, x1, y1, canvas, "•")
        
        if right_points:
            smooth_right = interpolate_points(right_points, right_cols)
            
            # Draw right curve
            for i in range(len(smooth_right) - 1):
                x0, y0 = int(smooth_right[i][0]), int(smooth_right[i][1])
                x1, y1 = int(smooth_right[i+1][0]), int(smooth_right[i+1][1])
                #draw_line(x0, y0, x1, y1, canvas, "•")
        
        # Draw filled area under curves (optional)
        for points, start_x in [(left_points, 0), (right_points, left_cols)]:
            for x, y in points:
                if start_x <= x < cols and 0 <= y < rows:
                    for fill_y in range(y, min(rows, y + 3)):  # Small fill
                        if 0 <= fill_y < rows:
                            canvas[fill_y][x] = "^" #"░"
        
    else:
        # Mono mode - single full-width curve
        # Use safe max value
        mono_vals = vals_left[:cols] if len(vals_left) >= cols else vals_left
        max_val = safe_max(mono_vals, 1)
        
        # Create curve points
        points = []
        for i in range(min(len(mono_vals), cols)):
            v = mono_vals[i]
            x = int((i / max(1, len(mono_vals)-1)) * (cols - 1))
            y = int((v / max_val) * (rows // 2))
            y_pos = mid - y
            y_pos = max(0, min(rows-1, y_pos))
            points.append((x, y_pos))
        
        # Only draw if we have points
        if points:
            # Create smooth curve
            smooth_points = interpolate_points(points, min(cols, len(points)*2))
            
            # Draw the curve
            for i in range(len(smooth_points) - 1):
                x0, y0 = int(smooth_points[i][0]), int(smooth_points[i][1])
                x1, y1 = int(smooth_points[i+1][0]), int(smooth_points[i+1][1])
                #draw_line(x0, y0, x1, y1, canvas, "•")
            
            # Draw filled area under curve (optional)
            for x, y in points:
                if 0 <= x < cols and 0 <= y < rows:
                    for fill_y in range(y, min(rows, y + 5)):  # Larger fill for mono
                        if 0 <= fill_y < rows:
                            canvas[fill_y][x] = "@" #"░"
    
    # Draw the canvas with optional frequency markers
    for y in range(rows):
        for x in range(cols):
            char = canvas[y][x]
            if char != " ":
                # Add color based on height (amplitude)
                height_ratio = (rows - y) / rows
                if height_ratio > 0.8:
                    color = PARTICLE_COLOR[0]  # Red for high amplitudes
                elif height_ratio > 0.5:
                    color = PARTICLE_COLOR[1]  # Yellow for medium amplitudes
                else:
                    color = PARTICLE_COLOR[2]  # Green for low amplitudes
                sys.stdout.write(f"\x1b[{color}m{char}\x1b[0m")
            else:
                # Optional: Draw frequency grid lines
                if y == mid:
                    sys.stdout.write("─")  # Center line
                elif x % 20 == 0:
                    sys.stdout.write("·")  # Vertical markers
                else:
                    sys.stdout.write(" ")
                #sys.stdout.write(" ")
        sys.stdout.write("\n")
    sys.stdout.flush()

def draw_binary(left_data, right_data=None):
    clear(); apply_colors()
    canvas = [[" "]*cols for _ in range(rows)]
    threshold = 0.5  # Adjustable threshold
    
    if right_data is not None:
        left_cols = cols // 2
        right_cols = cols // 2
        
        step_left = max(1, len(left_data)//left_cols)
        step_right = max(1, len(right_data)//right_cols)
        
        for x in range(left_cols):
            if abs(left_data[x*step_left]) > threshold:
                for y in range(rows):
                    canvas[y][x] = CHAR  # Or "█" for filled
        
        for x in range(right_cols):
            col_pos = left_cols + x
            if abs(right_data[x*step_right]) > threshold:
                for y in range(rows):
                    canvas[y][col_pos] = CHAR
    else:
        step = max(1, len(left_data)//cols)
        for x in range(cols):
            if abs(left_data[x*step]) > threshold:
                for y in range(rows):
                    canvas[y][x] = CHAR
    
    for r in canvas: sys.stdout.write("".join(r) + "\n")
    sys.stdout.flush()

def callback(indata, frames, time_, status):
    # Process stereo input (both channels)
    global psychedelic, fg_i, psychcount, psychdelay, spectrogram_history, reverse, PARTICLE_COLOR
    
    VERTICAL_BINS = rows  # Use screen height for vertical resolution

    if indata.shape[1] >= 2:
        left_data = indata[:, 0]
        right_data = indata[:, 1]
        left_fft = np.abs(np.fft.rfft(left_data))
        right_fft = np.abs(np.fft.rfft(right_data))
        
        left_bins = np.array_split(left_fft, BARS)
        right_bins = np.array_split(right_fft, BARS)
        
        # Calculate values for both channels
        vals_left = [int(np.log10(b.mean()+1)*MAX_H) for b in left_bins]
        vals_right = [int(np.log10(b.mean()+1)*MAX_H) for b in right_bins]
        vals_left_centered = [int(np.log10(b.mean()+1)*(MAX_H//2)) for b in left_bins]
        vals_right_centered = [int(np.log10(b.mean()+1)*(MAX_H//2)) for b in right_bins]
        
        # For mono compatibility in some modes
        avg_fft = (left_fft + right_fft) / 2
        bins_avg = np.array_split(avg_fft, BARS)
        vals_avg = [int(np.log10(b.mean()+1)*MAX_H) for b in bins_avg]
        vals_avg_centered = [int(np.log10(b.mean()+1)*(MAX_H//2)) for b in bins_avg]
    else:
        # Fallback to mono if only one channel
        data = indata[:, 0]
        left_data = data
        right_data = None
        fft = np.abs(np.fft.rfft(data))
        bins = np.array_split(fft, BARS)
        vals_left = [int(np.log10(b.mean()+1)*MAX_H) for b in bins]
        vals_left_centered = [int(np.log10(b.mean()+1)*(MAX_H//2)) for b in bins]
        vals_avg = vals_left
        vals_avg_centered = vals_left_centered
        
    if psychedelic:
        psychcount += 1
        if psychcount == psychdelay:
            fg_i = (fg_i + 1) % len(fg_colors)
            psychcount = 0
            if fg_i == 0:
                PARTICLE_COLOR = (32,33,31)
            elif fg_i == 1:
                PARTICLE_COLOR = (31,35,34)
            elif fg_i == 3:
                PARTICLE_COLOR = (30,33,31)
            elif fg_i == 4:
                PARTICLE_COLOR = (36,34,30)
            elif fg_i == 5:
                PARTICLE_COLOR = (33,33,36)
            elif fg_i == 6:
                PARTICLE_COLOR = (33,31,35)
            elif fg_i == 7:
                PARTICLE_COLOR = (34,36,33)

    if mode == 1:
        # Normal bars - stereo or mono
        if reverse:
            if indata.shape[1] >= 2:
                draw_normal_reverse(vals_left, vals_right)
            else:
                draw_normal_reverse(vals_left)
            return
        else:
            if indata.shape[1] >= 2:
                draw_normal(vals_left, vals_right)
            else:
                draw_normal(vals_left)
            return
    elif mode == 2:
        # Centered bars - stereo or mono
        if reverse:
            if indata.shape[1] >= 2:
                draw_centered_reverse(vals_left_centered, vals_right_centered)
            else:
                draw_centered_reverse(vals_left_centered)
            return
        else:
            if indata.shape[1] >= 2:
                draw_centered(vals_left_centered, vals_right_centered)
            else:
                draw_centered(vals_left_centered)
    elif mode == 3:
        # Circle mode - stereo or mono
        r = int(np.mean(np.abs(left_data))*min(cols, rows)*1.5)
        draw_circle(r)
    elif mode == 4:
        # Waveform mode - stereo or mono
        if indata.shape[1] >= 2:
            draw_waveform(left_data, right_data)
        else:
            draw_waveform(left_data)
    elif mode == 5:
        # Bar waveform mode - stereo or mono
        if indata.shape[1] >= 2:
            draw_bar_waveform(left_data, right_data)
        else:
            draw_bar_waveform(left_data)
    elif mode == 6:
        # Dual circle mode - stereo or mono
        if indata.shape[1] >= 2:
            left_r = int(np.mean(np.abs(left_data))*min(cols//2, rows)*1.5)
            right_r = int(np.mean(np.abs(right_data))*min(cols//2, rows)*1.5)
            draw_circle_dual(left_r, right_r)
        else:
            r = int(np.mean(np.abs(left_data))*min(cols//2, rows)*1.5)
            draw_circle_dual(r, r)
    elif mode == 7:
        # We need the full FFT data, not just binned values
        if indata.shape[1] >= 2:
            # Use the binned values you already have
            draw_particles_basic(vals_left, vals_right)
        else:
            draw_particles_basic(vals_left)
    elif mode == 9:
        # Centered bars - stereo or mono
        if indata.shape[1] >= 2:
            draw_lissajous(left_data, right_data)  # Pass raw audio, not FFT values
        else:
            draw_lissajous(left_data)  # Pass raw audio, not FFT values
    elif mode == 8:
        if indata.shape[1] >= 2:
            draw_binary(left_data, right_data)  # Pass raw audio, not FFT values
        else:
            draw_binary(left_data)  # Pass raw audio, not FFT values
    elif mode == 0:  # EQ Curve mode (assuming this is mode 9)
        if indata.shape[1] >= 2:
            # Scale values for better visual range
            eq_left = [int(v * 1.5) for v in vals_left[:cols//2]]
            eq_right = [int(v * 1.5) for v in vals_right[:cols//2]]
            draw_eq_curve(eq_left, eq_right)
            # Or use the simple version:
            # draw_eq_curve_simple(eq_left, eq_right)
        else:
            eq_mono = [int(v * 1.5) for v in vals_left[:cols]]
            draw_eq_curve(eq_mono)
            # draw_eq_curve_simple(eq_mono)




print("\x1b[2J\x1b[H\x1b[0m", end="")
threading.Thread(target=key_listener, daemon=True).start()

# Change to 2 channels for stereo input
with sd.InputStream(channels=2, samplerate=RATE, blocksize=BLOCK, callback=callback):
    while running:
        time.sleep(0.1)

sys.stdout.write("\x1b[0m")
