import sys
import numpy as np
import pyaudio
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import colorsys

class AudioEqualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # 1. Setup UI
        self.view = pg.GraphicsLayoutWidget(title="🎵 Premium Audio Visualizer")
        self.setCentralWidget(self.view)
        self.setWindowTitle('🎵 Premium Audio Visualizer')
        self.resize(1200, 700)
        
        # Set dark background with style
        self.view.setBackground((10, 10, 15))
        
        # 2. Setup Plot with symmetrical layout
        self.plot = self.view.addPlot(title="<span style='color: #00ffff; font-size: 16pt; font-weight: bold;'>✨ Frequency Spectrum ✨</span>")
        self.plot.setYRange(-50, 50, padding=0.05)  # Symmetrical: -50 to +50
        self.plot.setXRange(0, 80, padding=0.02)
        self.plot.setLabel('left', '<span style="color: #00ffff;">Level (dB)</span>')
        self.plot.setLabel('bottom', '<span style="color: #00ffff;">Frequency Band</span>')
        self.plot.showGrid(x=False, y=True, alpha=0.2)
        self.plot.getAxis('left').setPen((0, 255, 255, 100))
        self.plot.getAxis('bottom').setPen((0, 255, 255, 100))
        
        # Create stunning mirror-effect bars
        self.num_bars = 80
        self.bars_top = []  # Bars growing upward
        self.bars_bottom = []  # Bars growing downward (mirror)
        self.peak_indicators_top = []  # Peak hold dots for top bars
        self.peak_indicators_bottom = []  # Peak hold dots for bottom bars (mirror)
        self.peak_values_top = np.zeros(self.num_bars)  # Track peak heights for top
        self.peak_values_bottom = np.zeros(self.num_bars)  # Track peak heights for bottom
        self.peak_fall_speed = 0.7  # How fast peaks fall (increased for more dynamic effect)
        
        for i in range(self.num_bars):
            # Dynamic color based on position (rainbow spectrum)
            hue = i / self.num_bars
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
            color = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), 200)
            
            # Create gradient brush for depth effect
            gradient = QtGui.QLinearGradient(0, 0, 0, 1)
            gradient.setColorAt(0, QtGui.QColor(*color))
            gradient.setColorAt(1, QtGui.QColor(color[0]//2, color[1]//2, color[2]//2, 180))
            brush = QtGui.QBrush(gradient)
            
            # Top bar (grows upward from center)
            bar_top = pg.BarGraphItem(x=[i], height=[0], width=0.85, brush=color, pen=pg.mkPen(color, width=1.5), y0=0)
            self.plot.addItem(bar_top)
            self.bars_top.append(bar_top)
            
            # Bottom bar (mirror - grows downward from center)
            bar_bottom = pg.BarGraphItem(x=[i], height=[0], width=0.85, brush=color, pen=pg.mkPen(color, width=1.5), y0=0)
            self.plot.addItem(bar_bottom)
            self.bars_bottom.append(bar_bottom)
            
            # Peak indicator for top bar (glowing dot)
            peak_dot_top = pg.ScatterPlotItem(pos=[[i, 0]], size=8, pen=pg.mkPen(None), 
                                         brush=pg.mkBrush(255, 255, 255, 220), 
                                         symbol='o', pxMode=True)
            self.plot.addItem(peak_dot_top)
            self.peak_indicators_top.append(peak_dot_top)
            
            # Peak indicator for bottom bar (mirror - falls upward from bottom)
            peak_dot_bottom = pg.ScatterPlotItem(pos=[[i, 0]], size=8, pen=pg.mkPen(None), 
                                         brush=pg.mkBrush(255, 255, 255, 220), 
                                         symbol='o', pxMode=True)
            self.plot.addItem(peak_dot_bottom)
            self.peak_indicators_bottom.append(peak_dot_bottom)
        
        # Add center line for reference
        center_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen((0, 255, 255, 80), width=2, style=QtCore.Qt.DashLine))
        self.plot.addItem(center_line)

        # 3. Setup Audio Capture
        self.CHUNK = 2048  # Increased for better frequency resolution
        self.RATE = 44100
        self.p = pyaudio.PyAudio()
        
        # Print available devices and try to find a suitable input
        print("\n=== Available Audio Input Devices ===")
        default_device = None
        cable_device = None
        b1_device = None  # Prioritize Voicemeeter B1 output

        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                name = info.get('name', '')
                print(f"Device {i}: {name} (Channels: {info['maxInputChannels']})")
                lname = name.lower()

                # First priority: Voicemeeter Out B1 (main output)
                if ('voicemeeter out b1' in lname or 'voicemeeter b1' in lname) and b1_device is None:
                    b1_device = i
                    print("  -> Found Voicemeeter B1 (main output)!")
                
                # Second priority: Other Voicemeeter output devices
                if ('voicemeeter' in lname or 'voice meeter' in lname) and 'out' in lname and cable_device is None:
                    cable_device = i
                    print("  -> Found Voicemeeter Output!")
                
                # Third priority: VB-Cable
                if ('cable' in lname and 'vb-audio' in lname) and cable_device is None and b1_device is None:
                    cable_device = i
                    print("  -> Found VB-Audio Cable!")

                # Skip "Sound Mapper" devices as they can cause issues; pick the first sane input
                if default_device is None and 'mapper' not in lname:
                    default_device = i

        # Allow user to pass a device index on the command line: python AudioEqualizer.py <device_index>
        input_device = None
        if len(sys.argv) > 1:
            try:
                input_device = int(sys.argv[1])
                print(f"\nUsing device index from command line: {input_device}")
            except Exception:
                print("\nWarning: invalid device index passed on CLI; falling back to auto-detect")

        # If no CLI override, prefer B1, then other virtual devices, then default device
        if input_device is None:
            if b1_device is not None:
                input_device = b1_device
                print("\n✓ Will use Voicemeeter B1 for system audio capture")
            elif cable_device is not None:
                input_device = cable_device
                print("\n✓ Will use Virtual Audio Device for system audio capture")
            elif default_device is not None:
                input_device = default_device
            else:
                # Try PyAudio's default input device as a last resort
                try:
                    default_info = self.p.get_default_input_device_info()
                    input_device = int(default_info.get('index', 0))
                except Exception:
                    input_device = 0

        # Report chosen device
        try:
            device_info = self.p.get_device_info_by_index(input_device)
            print(f"\nUsing Device {input_device}: {device_info.get('name', 'Unknown')}")
        except Exception as e:
            print(f"\nWarning: unable to query device {input_device}: {e}")
            print("Falling back to device 0")
            input_device = 0
            device_info = self.p.get_device_info_by_index(input_device)
        
        # Determine number of channels
        max_channels = min(device_info['maxInputChannels'], 2)
        self.channels = max_channels if max_channels > 0 else 1
        
        print(f"Channels: {self.channels}, Sample Rate: {self.RATE} Hz\n")
        
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.RATE,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=self.CHUNK
            )
            print(f"✓ Successfully opened audio stream!\n")
        except Exception as e:
            print(f"✗ Error opening audio device: {e}\n")
            raise

        # 4. Refresh Timer (to update the bars)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(25)  # ~40 FPS for smoother animation
        
        # Enhanced visualization settings
        self.noise_threshold = 0
        self.sensitivity = 1.2  # Increased for more dramatic effect
        self.peak_hold = 0
        
        # Animation smoothing with velocity for physics-based movement
        self.bar_velocities = np.zeros(self.num_bars)
        self.bar_accelerations = np.zeros(self.num_bars)
        self.current_heights = np.zeros(self.num_bars)
        
        # Bass reactivity for background pulse
        self.bass_intensity = 0
        self.bass_smoothing = 0.85

    def update(self):
        try:
            # Read raw data and convert to numpy array
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            data_int = np.frombuffer(data, dtype=np.int16)
            
            # Convert stereo to mono if needed
            if self.channels == 2:
                data_int = data_int.reshape(-1, 2).mean(axis=1).astype(np.int16)
            
            # Check if there's actual audio signal (not just digital noise)
            audio_level = np.abs(data_int).mean()
            
            # Debug: Print audio level every ~1 second (40 frames at 40 FPS)
            if not hasattr(self, '_frame_count'):
                self._frame_count = 0
            self._frame_count += 1
            if self._frame_count % 40 == 0:
                print(f"Audio level: {audio_level:.1f}", end='\r')
            
            if audio_level < 30:
                # Smooth fade out with physics
                self.current_heights *= 0.85
                self.bar_velocities *= 0.9
                for i in range(self.num_bars):
                    self.bars_top[i].setOpts(height=self.current_heights[i])
                    self.bars_bottom[i].setOpts(height=-self.current_heights[i])
                    
                    # Update peak indicators (top falls down, bottom falls up)
                    self.peak_values_top[i] = max(self.peak_values_top[i] - self.peak_fall_speed, 0)
                    self.peak_indicators_top[i].setData(pos=[[i, self.peak_values_top[i]]])
                    
                    self.peak_values_bottom[i] = min(self.peak_values_bottom[i] + self.peak_fall_speed, 0)
                    self.peak_indicators_bottom[i].setData(pos=[[i, self.peak_values_bottom[i]]])
                
                # Fade bass effect
                self.bass_intensity *= 0.9
                self._update_background()
                return
            
            # Apply Hamming window to reduce spectral leakage
            window = np.hamming(len(data_int))
            data_windowed = data_int * window
            
            # Fast Fourier Transform (FFT)
            fft_data = np.abs(np.fft.rfft(data_windowed))
            
            # Get frequency bins and select range up to 8kHz for better visualization
            freqs = np.fft.rfftfreq(len(data_windowed), 1/self.RATE)
            max_freq = 8000  # Focus on audible range
            freq_indices = np.where(freqs <= max_freq)[0]
            
            # Downsample to match number of bars
            samples_per_bar = len(freq_indices) // self.num_bars
            if samples_per_bar > 0:
                fft_bars = []
                for i in range(self.num_bars):
                    start_idx = i * samples_per_bar
                    end_idx = start_idx + samples_per_bar
                    if end_idx < len(freq_indices):
                        # Average the FFT values in this range
                        bar_value = np.mean(fft_data[freq_indices[start_idx:end_idx]])
                        fft_bars.append(bar_value)
                    else:
                        fft_bars.append(0)
                
                # Convert to decibel-like scale with better sensitivity
                fft_bars = np.array(fft_bars)
                fft_bars = np.log10(fft_bars + 1) * 20  # Log scale
                
                # Apply noise gate - filter out background noise
                fft_bars = np.where(fft_bars < self.noise_threshold, 0, fft_bars - self.noise_threshold)
                
                # Track peak for adaptive normalization (slowly decay peak over time)
                current_max = np.max(fft_bars)
                if current_max > self.peak_hold:
                    self.peak_hold = current_max
                else:
                    self.peak_hold = self.peak_hold * 0.995  # Slowly decay peak
                
                # Normalize using peak hold for more stable visualization
                if self.peak_hold > 1:
                    fft_bars = (fft_bars / self.peak_hold) * 60 * self.sensitivity  # Scale to 60 dB range
                else:
                    fft_bars = fft_bars * self.sensitivity
                
                # Scale to 0-50 range for symmetrical display
                fft_bars = np.clip(fft_bars, 0, 60) * (50/60)
                
                # Calculate bass intensity (average of first 10 bars for low frequencies)
                bass_level = np.mean(fft_bars[:10])
                self.bass_intensity = self.bass_intensity * self.bass_smoothing + bass_level * (1 - self.bass_smoothing)
                
                # Update bars with physics-based smooth animation
                for i in range(self.num_bars):
                    target_height = fft_bars[i]
                    
                    # Calculate acceleration towards target (spring physics)
                    self.bar_accelerations[i] = (target_height - self.current_heights[i]) * 0.5
                    
                    # Update velocity with damping
                    self.bar_velocities[i] = self.bar_velocities[i] * 0.6 + self.bar_accelerations[i]
                    
                    # Update position
                    self.current_heights[i] += self.bar_velocities[i]
                    self.current_heights[i] = max(0, self.current_heights[i])  # Clamp to positive
                    
                    # Dynamic color intensity based on height
                    intensity = min(1.0, self.current_heights[i] / 40)
                    hue = i / self.num_bars
                    rgb = colorsys.hsv_to_rgb(hue, 0.8 + intensity * 0.2, 0.7 + intensity * 0.3)
                    glow = 150 + int(intensity * 105)  # Brightness glow effect
                    color = (int(rgb[0] * glow), int(rgb[1] * glow), int(rgb[2] * glow), 200)
                    
                    # Add extra glow to bass frequencies (first 20 bars)
                    if i < 20:
                        bass_boost = 1.0 + (self.bass_intensity / 50) * 0.5
                        color = (min(255, int(color[0] * bass_boost)), 
                                min(255, int(color[1] * bass_boost)), 
                                min(255, int(color[2] * bass_boost)), 220)
                    
                    # Update top and bottom bars (mirror effect)
                    self.bars_top[i].setOpts(height=self.current_heights[i], brush=color, pen=pg.mkPen(color, width=2))
                    self.bars_bottom[i].setOpts(height=-self.current_heights[i], brush=color, pen=pg.mkPen(color, width=2))
                    
                    # Update peak indicators with smooth fall (top and bottom mirror)
                    # Top peak indicator (falls downward from peak)
                    if self.current_heights[i] > self.peak_values_top[i]:
                        self.peak_values_top[i] = self.current_heights[i]
                    else:
                        self.peak_values_top[i] = max(self.peak_values_top[i] - self.peak_fall_speed, 0)
                    
                    # Bottom peak indicator (falls upward from bottom peak)
                    if -self.current_heights[i] < self.peak_values_bottom[i]:
                        self.peak_values_bottom[i] = -self.current_heights[i]
                    else:
                        self.peak_values_bottom[i] = min(self.peak_values_bottom[i] + self.peak_fall_speed, 0)
                    
                    # Make peak dots glow more when near extremes
                    peak_alpha = 150 + int((self.peak_values_top[i] / 50) * 105)
                    dot_size = 6 + int(intensity * 4)
                    
                    # Update top peak dot
                    self.peak_indicators_top[i].setData(pos=[[i, self.peak_values_top[i]]], 
                                                   brush=pg.mkBrush(255, 255, 255, peak_alpha),
                                                   size=dot_size)
                    
                    # Update bottom peak dot (mirror)
                    self.peak_indicators_bottom[i].setData(pos=[[i, self.peak_values_bottom[i]]], 
                                                   brush=pg.mkBrush(255, 255, 255, peak_alpha),
                                                   size=dot_size)
                
                # Update background based on bass
                self._update_background()
                
        except Exception as e:
            print(f"Error reading audio: {e}")
    
    def _update_background(self):
        """Update background color based on bass intensity for immersive effect"""
        # Calculate pulsing background color
        bass_factor = min(1.0, self.bass_intensity / 30)
        
        # Subtle color shift from dark blue to purple/red with bass
        r = int(10 + bass_factor * 30)
        g = int(10 + bass_factor * 5)
        b = int(15 + bass_factor * 25)
        
        self.view.setBackground((r, g, b))

    def closeEvent(self, event):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = AudioEqualizer()
    window.show()
    sys.exit(app.exec_())
