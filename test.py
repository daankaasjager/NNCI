import math

def calculate_frequency_response(h, omega):
    real_part = 0
    imag_part = 0
    for n, h_n in enumerate(h):
        real_part += h_n * math.cos(-omega * n)
        imag_part += h_n * math.sin(-omega * n)
    return real_part, imag_part

def main():
    # Example input
    h1 = [0, 2, 0, -2]  # Impulse response of the first FIR filter
    h2 = [1, 0, 1, 2]     # Impulse response of the second FIR filter
    Ax, omega, phix = 3, 2.7489, 2.75 # Parameters of the input signal

    # Calculate frequency response of each filter
    H1_real, H1_imag = calculate_frequency_response(h1, omega)
    H2_real, H2_imag = calculate_frequency_response(h2, omega)

    # Combine the responses
    combined_real = H1_real + H2_real
    combined_imag = H1_imag + H2_imag

    # Calculate output amplitude and phase
    Ay = Ax * math.sqrt(combined_real**2 + combined_imag**2)
    phiy = math.atan2(combined_imag, combined_real) + phix

    # Round values to two decimal places
    Ay = round(Ay, 2)
    phiy = round(phiy, 2)
    omega = round(omega, 2)

    # Ensure phase is within the specified range
    phiy = (phiy + math.pi) % (2 * math.pi) - math.pi

    # Output
    if Ay == 0:
        print("y[n]=0.00")
    else:
        print(f"y[n]={Ay}cos({omega}*n+{phiy})")

if __name__ == "__main__":
    main()
