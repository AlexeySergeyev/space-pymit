import csv

def generate_csv():
    # we'll read a few lines from damit/test_lcs_abs and write them as a valid CSV.
    input_txt = "damit/test_lcs_abs"
    output_csv = "test_lcs.csv"
    
    lines = []
    with open(input_txt, 'r') as f:
        # read first curve
        num_curves = int(f.readline().strip())
        curve_header = f.readline().strip().split()
        num_points = int(curve_header[0])
        is_abs = int(curve_header[1]) # 1 is abs, meaning is_relative = 0
        is_relative = 0 if is_abs == 1 else 1
        
        for _ in range(num_points):
            lines.append(f.readline().strip())

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['curve_id', 'is_relative', 'jd', 'brightness', 'sun_x', 'sun_y', 'sun_z', 'earth_x', 'earth_y', 'earth_z'])
        for line in lines:
            parts = line.split()
            writer.writerow(['1', is_relative] + parts)
            
    print(f"Generated {output_csv}")

if __name__ == "__main__":
    generate_csv()
