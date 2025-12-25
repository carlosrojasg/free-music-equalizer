import sys
import numpy as np
import pyaudio
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import colorsys

class AudioEqualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # 1. Setup UI
        self.view = pg.GraphicsLayoutWidget(title="Python Real-time Equalizer")
        self.setCentralWidget(self.view)
        self.setWindowTitle('Real-time Audio Visualizer')
        self.resize(1000, 500)
        
        # 2. Setup Plot
        self.plot = self.view.addPlot(title="Frequency Spectrum")
        self.plot.setYRange(0, 70, padding=0)  # Set to 70 to fill more of the screen
        self.plot.setXRange(0, 80, padding=0.02)
        self.plot.setLabel('left', 'Intensity (dB)')
        self.plot.setLabel('bottom', 'Frequency Band')
        self.plot.showGrid(x=False, y=True, alpha=0.3)
        
        # Create colorful bars using rainbow gradient
        self.num_bars = 80
        self.bars = []
        for i in range(self.num_bars):
            # Rainbow gradient using HSV color space
            hue = i / self.num_bars  # 0 to 1
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            color = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            bar = pg.BarGraphItem(x=[i], height=[0], width=0.9, brush=color)
            self.plot.addItem(bar)
            self.bars.append(bar)

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
        self.timer.start(30) # ~30 FPS
        
        # Noise gate and sensitivity settings
        self.noise_threshold = 8  # Adjust this value (10-30) - higher = less sensitive to noise
        self.sensitivity = 0.9  # Controls overall bar height (0.1 = quiet, 1.0 = loud)
        self.peak_hold = 0  # Tracks the highest FFT value seen for better normalization

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
            
            # Debug: Print audio level every ~1 second (30 frames at 30 FPS)
            if not hasattr(self, '_frame_count'):
                self._frame_count = 0
            self._frame_count += 1
            if self._frame_count % 30 == 0:
                print(f"Audio level: {audio_level:.1f}", end='\r')
            
            if audio_level < 30:  # If audio level is too low, don't update bars
                # Fade out bars gradually
                for i in range(self.num_bars):
                    current_height_data = self.bars[i].opts['height']
                    if isinstance(current_height_data, (list, np.ndarray)) and len(current_height_data) > 0:
                        current_height = current_height_data[0]
                    elif isinstance(current_height_data, (int, float, np.number)):
                        current_height = float(current_height_data)
                    else:
                        current_height = 0
                    
                    # Fade to zero
                    fade_height = current_height * 0.7
                    self.bars[i].setOpts(height=fade_height)
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
                    fft_bars = (fft_bars / self.peak_hold) * 70 * self.sensitivity  # Scale to 70
                else:
                    fft_bars = fft_bars * self.sensitivity
                
                # Clip to 0-70 range
                fft_bars = np.clip(fft_bars, 0, 70)
                
                # Update bar heights with smoothing
                for i in range(self.num_bars):
                    # Get current height safely
                    current_height_data = self.bars[i].opts['height']
                    if isinstance(current_height_data, (list, np.ndarray)) and len(current_height_data) > 0:
                        current_height = current_height_data[0]
                    elif isinstance(current_height_data, (int, float, np.number)):
                        current_height = float(current_height_data)
                    else:
                        current_height = 0
                    
                    new_height = fft_bars[i]
                    # Smooth transition
                    smooth_height = current_height * 0.5 + new_height * 0.5
                    self.bars[i].setOpts(height=smooth_height)
                
        except Exception as e:
            print(f"Error reading audio: {e}")

    def closeEvent(self, event):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = AudioEqualizer()
    window.show()
    sys.exit(app.exec_())
