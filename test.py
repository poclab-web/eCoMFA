import psi4

def test_psi4_energy_calculation():
    # Set molecule geometry (water molecule)
    h2o = psi4.geometry("""
    O
    H 1 1.0
    H 1 1.0 2 104.5
    """)
    
    # Set computational options
    psi4.set_options({'basis': 'sto-3g', 'scf_type': 'pk'})
    
    # Perform energy calculation
    energy = psi4.energy('scf/sto-3g')
    
    # Expected energy value (This is just an example value)
    expected_energy = -74.96
    
    # Check if the calculated energy is within an acceptable range of the expected value
    assert abs(energy - expected_energy) < 1e-2, f"Energy calculation failed. Expected: {expected_energy}, Got: {energy}"
    
    print("Psi4 energy calculation test passed.")

# Run the test
test_psi4_energy_calculation()
