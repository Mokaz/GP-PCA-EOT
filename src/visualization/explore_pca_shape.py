import sys
import numpy as np
import panel as pn
import plotly.graph_objects as go
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
SRC_ROOT = FILE_PATH.parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.geometry_utils import compute_estimated_shape_from_params

pn.extension('plotly')

class ShapeExplorer:
    def __init__(self, pca_path: str):
        # 1. Load PCA Data
        if not Path(pca_path).exists():
            raise FileNotFoundError(f"Could not find PCA file at: {pca_path}")
            
        data = np.load(pca_path)
        self.eigenvectors = data['eigenvectors'].real 
        self.mean_coeffs = data['mean']
        self.num_pca_comps = self.eigenvectors.shape[1]
        self.n_fourier_dim = self.mean_coeffs.shape[0]
        self.angles = np.linspace(0, 2*np.pi, 200)

        # 2. Define GUI Widgets
        self.slider_L = pn.widgets.FloatSlider(
            name='Length (L)', start=5.0, end=150.0, step=1.0, value=50.0, 
            sizing_mode='stretch_width'
        )
        self.slider_W = pn.widgets.FloatSlider(
            name='Width (W)', start=2.0, end=50.0, step=0.5, value=20.0, 
            sizing_mode='stretch_width'
        )
        self.rotate_toggle = pn.widgets.Checkbox(
            name='Rotate 90°', value=False, 
            sizing_mode='stretch_width'
        )
        self.reset_btn = pn.widgets.Button(
            name='Reset Coefficients', button_type='primary', 
            sizing_mode='stretch_width'
        )
        self.reset_btn.on_click(self.reset_coefficients)

        self.coeff_input = pn.widgets.TextInput(
            name='Paste Coefficients (comma separated)',
            placeholder='e.g. -1.17, 0.16, 0.06, 0.07',
            sizing_mode='stretch_width'
        )
        self.set_coeff_btn = pn.widgets.Button(
            name='Set Coefficients', button_type='success',
            sizing_mode='stretch_width'
        )
        self.set_coeff_btn.on_click(self.set_coefficients_from_text)

        self.pca_sliders = []
        for i in range(self.num_pca_comps):
            s = pn.widgets.FloatSlider(
                name=f'PCA Coeff {i}', start=-100.0, end=100.0, step=0.1, value=0.0,
                sizing_mode='stretch_width'
            )
            self.pca_sliders.append(s)

        self.plot_pane = pn.pane.Plotly(
            sizing_mode='stretch_both', 
            config={'responsive': True}
        )

        pn.bind(self.update_plot, 
                self.slider_L, 
                self.slider_W, 
                self.rotate_toggle, 
                *self.pca_sliders, 
                watch=True)

        self.update_plot(self.slider_L.value, self.slider_W.value, self.rotate_toggle.value, *[s.value for s in self.pca_sliders])

    def reset_coefficients(self, event=None):
        for slider in self.pca_sliders:
            slider.value = 0.0

    def set_coefficients_from_text(self, event=None):
        """Parses the text input and updates sliders."""
        text = self.coeff_input.value
        if not text:
            return
        
        try:
            # Remove brackets if present and split by comma
            clean_text = text.replace('[', '').replace(']', '').replace('np.array(', '').replace(')', '')
            values = [float(x.strip()) for x in clean_text.split(',')]
            
            for i, val in enumerate(values):
                if i < len(self.pca_sliders):
                    self.pca_sliders[i].value = val
                    
        except ValueError:
            print("Invalid input format. Please use comma-separated numbers.")

    def update_plot(self, L, W, rotate_90, *pca_coeffs):
        """
        Callback function that updates the EXISTING plot pane.
        """
        coeffs_array = np.array(pca_coeffs)
        yaw = np.pi / 2 if rotate_90 else 0.0

        est_shape_x, est_shape_y = compute_estimated_shape_from_params(
            0.0, 0.0, yaw, L, W, 
            coeffs_array, self.mean_coeffs, self.eigenvectors, 
            self.angles, self.n_fourier_dim
        )

        fig = go.Figure()

        # 1. GP-PCA Shape
        fig.add_trace(go.Scatter(
            x=est_shape_y, 
            y=est_shape_x, 
            mode='lines', 
            name='PCA Shape',
            line=dict(color='royalblue', width=3),
            fill='toself',
            fillcolor='rgba(65, 105, 225, 0.2)' 
        ))

        # 2. Reference Box
        box_local_x = np.array([L/2, L/2, -L/2, -L/2, L/2])
        box_local_y = np.array([-W/2, W/2, W/2, -W/2, -W/2])
        
        c, s = np.cos(yaw), np.sin(yaw)
        box_rot_x = c * box_local_x - s * box_local_y
        box_rot_y = s * box_local_x + c * box_local_y

        fig.add_trace(go.Scatter(
            x=box_rot_y, 
            y=box_rot_x, 
            mode='lines',
            name='Reference Box',
            line=dict(color='gray', dash='dash')
        ))

        fig.update_layout(
            title="Real-time Shape Exploration",
            autosize=True,
            xaxis=dict(range=[-60, 60], constrain='domain', title="Width / Local Y [m]"), 
            yaxis=dict(range=[-80, 80], scaleanchor="x", scaleratio=1, title="Length / Local X [m]"), 
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=20, r=20, t=40, b=20)
        )

        self.plot_pane.object = fig 

    def view(self):
        sidebar = pn.Column(
            pn.pane.Markdown("### Geometry Parameters"),
            self.slider_L,
            self.slider_W,
            self.rotate_toggle,
            pn.layout.Divider(),
            pn.pane.Markdown("### PCA Coefficients"),
            self.coeff_input,
            self.set_coeff_btn,
            self.reset_btn,
            pn.Spacer(height=10),
            *self.pca_sliders,
            sizing_mode='stretch_width' 
        )
        
        main_area = pn.Column(
            self.plot_pane,
            sizing_mode='stretch_both'
        )
        
        return pn.template.FastListTemplate(
            title="PCA Shape Explorer",
            sidebar=[sidebar],
            main=[main_area],
            sidebar_width=450
        )

if __name__ == "__main__":
    PCA_FILE_PATH = PROJECT_ROOT / "data" / "input_parameters" / "FourierPCAParameters_scaled.npz" 
    
    if not PCA_FILE_PATH.exists():
        print(f"WARNING: PCA file not found at {PCA_FILE_PATH}")
    else:
        explorer = ShapeExplorer(str(PCA_FILE_PATH))
        explorer.view().show()