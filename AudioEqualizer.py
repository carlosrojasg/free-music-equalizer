import sys
import numpy as np
import pyaudio
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

class AudioEqualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # 1. Setup UI
        self.view = pg.GraphicsLayoutWidget(title="Python Real-time Equalizer")
        self.setCentralWidget(self.view)
        self.setWindowTitle('Real-time Audio Visualizer')
        self.resize(800, 400)
        
        # 2. Setup Plot
        self.plot = self.view.addPlot(title="Frequency Spectrum")
        self.plot.setYRange(0, 50, padding=0)
        self.plot.setXRange(0, 100, padding=0) # Focus on low-mid frequencies
        
        # Create colorful bars using a gradient
        self.num_bars = 50
        self.bars = []
        for i in range(self.num_bars):
            # Gradient from Blue to Red
            color = (int(255 * (i/self.num_bars)), 100, 255 - int(255 * (i/self.num_bars)))
            bar = pg.BarGraphItem(x=[i], height=[0], width=0.8, brush=color)
            self.plot.addItem(bar)
            self.bars.append(bar)

        # 3. Setup Audio Capture
        self.CHUNK = 1024 
        self.RATE = 44100
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        # 4. Refresh Timer (to update the bars)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30) # ~30 FPS

    def update(self):
        try:
            # Read raw data and convert to numpy array
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            data_int = np.frombuffer(data, dtype=np.int16)
            
            # Fast Fourier Transform (FFT)
            fft_data = np.abs(np.fft.fft(data_int))[:self.num_bars]
            fft_data = np.log10(fft_data + 1) * 10 # Convert to decibel-like scale
            
            # Update bar heights
            for i in range(self.num_bars):
                self.bars[i].setOpts(height=fft_data[i])
                
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
