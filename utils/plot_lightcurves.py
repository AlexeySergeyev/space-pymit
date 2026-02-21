import matplotlib.pyplot as plt
from pathlib import Path

def plot_lcs_offline():
    # Adjust paths based on whether run from project root or utils dir
    base_dir = Path(__file__).parent.parent
    
    input_file = base_dir / 'damit' / 'test_lcs_abs'
    output_file = base_dir / 'pipeline_output' / 'test_asteroid_lcs.txt'
    assets_dir = base_dir / 'assets'
    
    assets_dir.mkdir(exist_ok=True)
    
    if not input_file.exists() or not output_file.exists():
        print(f"Required files not found. Ensure pipeline has been run first.")
        return

    with open(input_file, 'r') as f_in, open(output_file, 'r') as f_out:
        n_curves = int(f_in.readline().strip())
        
        plt.figure(figsize=(10, 6))
        
        for i in range(n_curves):
            header = f_in.readline().split()
            n_pts = int(header[0])
            # is_abs = int(header[1]) # not used in logic
            
            x = []
            y_obs = []
            y_mod = []
            
            for _ in range(n_pts):
                parts = f_in.readline().split()
                # Relative time to start of curve for plotting
                if not x:
                    t0 = float(parts[0])
                x.append(float(parts[0]) - t0)
                y_obs.append(float(parts[1]))
                y_mod.append(float(f_out.readline().strip()))
            
            if i < 3:  # Plot first 3 curves side by side or overlaid
                plt.plot(x, y_obs, 'o', label=f'Observed Curve {i+1}')
                plt.plot(x, y_mod, '-', label=f'Modeled Curve {i+1}')
                
        plt.xlabel('Time (Days from curve start)')
        plt.ylabel('Brightness')
        plt.title('Observed vs Modeled Light Curves (Sample)')
        plt.legend()
        
        save_path = assets_dir / 'lightcurves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {save_path}")

if __name__ == '__main__':
    plot_lcs_offline()
