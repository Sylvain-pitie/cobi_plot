#!/usr/bin/env python3
"""
Ultra-simple COBI script with plotting - One single command
Usage: python quick_cobi.py [--plot] [atom1] [atom2] [distances...]
"""

import sys
import os
from pymatgen.io.lobster import Icohplist
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.plotter import CohpPlotter
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.core import Structure
import numpy as np
import matplotlib.pyplot as plt
import argparse
import palettable


def quick_icobi(emin, emax, atom1, atom2, min_dist=0.0, max_dist=None, radius=None, plot=False):
    """
    Quickly computes the average ICOBI between two atom types

    Args:
        atom1, atom2: Atom types (e.g., "Pb", "N")
        min_dist: Minimum distance in Ångström (default: 0.0)
        max_dist: Maximum distance in Ångström
        radius: Search radius (for compatibility, equivalent to max_dist)
        plot: If True, plots COBI vs energy curves

    Returns:
        Dictionary with results
    """

    # Handle compatibility with older versions
    if radius is not None and max_dist is None:
        max_dist = radius
    elif max_dist is None:
        max_dist = 5.0  # Default value

    print(f"🔍 Searching for {atom1}-{atom2} bonds between {min_dist:.2f} and {max_dist:.2f} Å")

    # Check for required files
    required_files = ["ICOBILIST.lobster", "COBICAR.lobster", "POSCAR"]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return None

    try:
        # Load data
        icohplist = Icohplist(filename="ICOBILIST.lobster")
        icohpcollection = icohplist.icohpcollection
        structure = Structure.from_file("POSCAR")

        # Load COBICAR for plotting
        if plot:
            try:
                cobicar = None
                # Method 1: Explicit format specification
                try:
                    cobicar = CompleteCohp.from_file(filename="COBICAR.lobster", fmt="LOBSTER", structure_file="POSCAR")
                    print("✓ COBICAR.lobster loaded (LOBSTER format), method1")
                except:
                    pass
                
                if cobicar is None:
                    print("⚠️ Could not load COBICAR.lobster with any method")
                    plot = False
                    
            except Exception as e:
                print(f"⚠️ Error while loading COBICAR.lobster: {e}")
                plot = False

        print(f"✓ Structure loaded: {structure.composition}")

        # Find sites for atom1
        sites_atom1 = []
        for i, site in enumerate(structure.sites):
            if str(site.specie) == atom1:
                sites_atom1.append(i + 1)  # LOBSTER uses 1-based indexing

        print(f"{atom1} sites: {sites_atom1}")
        sites_atom2 = []
        for i, site in enumerate(structure.sites):
            if str(site.specie) == atom2:
                sites_atom2.append(i + 1)  # LOBSTER uses 1-based indexing

        print(f"{atom2} sites: {sites_atom2}")

        # Collect all bonds
        all_labels = []
        all_values = []
        seen_labels = set()  # To avoid duplicates
########################################################################"
        for site_idx in sites_atom1:
            site_bonds = icohpcollection.get_icohp_dict_of_site(
                site=int((site_idx-1)),
                minbondlength=float(min_dist),
                maxbondlength=float(max_dist)
            )


            for label, icohp_obj in site_bonds.items():
                if label in seen_labels:
                    continue  # Skip duplicates

                bond_info = str(icohp_obj)
                # Check if the bond involves atom1 and atom2
                if str(atom1) in bond_info and str(atom2) in bond_info:
                    icobi_value = icohpcollection.get_icohp_by_label(label)
                    all_labels.append(label)
                    all_values.append(icobi_value)
                    seen_labels.add(label)
                    print(f"  Bond {label}: ICOBI = {icobi_value:.6f}")
##########################################################################
        for site_idx in sites_atom2:
            site_bonds = icohpcollection.get_icohp_dict_of_site(
                site=int(site_idx-1),
                minbondlength=float(min_dist),
                maxbondlength=float(max_dist)
            )


            for label, icohp_obj in site_bonds.items():
                if label in seen_labels:
                    continue  # Skip duplicates

                bond_info = str(icohp_obj)
                # Check if the bond involves atom1 and atom2
                if str(atom1) in bond_info and str(atom2) in bond_info:
                    icobi_value = icohpcollection.get_icohp_by_label(label)
                    all_labels.append(label)
                    all_values.append(icobi_value)
                    seen_labels.add(label)
                    print(f"  Bond {label}: ICOBI = {icobi_value:.6f}")
        

        if not all_values:
            print(f"❌ No {atom1}-{atom2} bonds found between {min_dist:.2f} and {max_dist:.2f} Å")
            return None

        # Compute statistics
        mean_icobi = np.mean(all_values)
        std_icobi = np.std(all_values)

        result = {
            "pair": f"{atom1}-{atom2}",
            "min_dist": min_dist,
            "max_dist": max_dist,
            "n_bonds": len(all_values),
            "mean_icobi": mean_icobi,
            "std_icobi": std_icobi,
            "values": all_values,
            "labels": all_labels
        }

        print(f"\n✅ RESULTS:")
        print(f"   Pair: {result['pair']}")
        print(f"   Distance: {result['min_dist']:.2f} - {result['max_dist']:.2f} Å")
        print(f"   Number of bonds: {result['n_bonds']}")
        print(f"   Mean ICOBI: {result['mean_icobi']:.6f}")
        print(f"   Std. Dev.: {result['std_icobi']:.6f}")
        print(f"   Min/Max: {min(all_values):.6f} / {max(all_values):.6f}")

        if plot:
            if 'cobicar' in locals() and cobicar is not None:
                plot_cobi_curves(cobicar, all_labels, atom1, atom2, result, emin, emax)
            else:
                print("❌ Cannot plot COBI curves - file not loaded")

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_cobi_curves(cobicar, labels, atom1, atom2, result, emin, emax):
    """
    Plot COBI vs energy curves for selected bonds

    Args:
        cobicar: CompleteCohp object
        labels: List of bond labels
        atom1, atom2: Atom types
        result: Dictionary of results
    """
    print(f"\n📊 Generating COBI plots...")

    # Matplotlib setup
    plt.style.use('default')

    all_energies = None
    all_cobi_data = []
    cp = CohpPlotter(are_cobis=True)
    legend = f"{atom1}-{atom2}"
    cp.add_cohp(legend, cobicar.get_summed_cohp_by_label_list(label_list=labels, divisor=len(labels), summed_spin_channels=True))

    # Color configuration
    ncolors = max(3, len(cp._cohps))
    ncolors = min(9, ncolors)
    colors = [(0.2157, 0.4941, 0.7216)] + palettable.colorbrewer.qualitative.Set1_9.mpl_colors[1:]
    integrated = False
    plot_negative = None
    fig, ax = plt.subplots(figsize=(8,6))
    allpts = []
    keys = list(cp._cohps)
    idx = key = None
    for idx, key in enumerate(keys):
        energies = cp._cohps[key]["energies"]
        populations = cp._cohps[key]["COHP"] if not integrated else cp._cohps[key]["ICOHP"]
        for spin in [Spin.up, Spin.down]:
            if spin in populations:
                x = energies
                y = -populations[spin] if plot_negative else populations[spin]
            allpts.extend(list(zip(x, y)))
            if spin == Spin.up:
                plt.plot(y, x, color=colors[idx % ncolors], linestyle="-", label=str(key), linewidth=1.5)
            else:
                plt.plot(y, x, color=colors[idx % ncolors], linestyle="--", linewidth=1)

    # Axis setup
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('COBI')
    ax.set_ylabel('Energy - Ef (eV)')
    ax.set_ylim(emin, emax)
    ax.legend(frameon=False, fontsize=16)

    # Save the figure
    filename = f"cobi_{atom1}_{atom2}_{result['min_dist']:.1f}-{result['max_dist']:.1f}A.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📁 Plot saved: {filename}")

def main():
    """Simple user interface"""
    print("🧪 Fast ICOBI Calculator with Plotting")
    print("=" * 45)

    # Parse arguments
    parser = argparse.ArgumentParser(description="ICOBI calculator with optional plotting")
    parser.add_argument('--plot', '-p', action='store_true', help='Show COBI vs energy plots')
    parser.add_argument('atoms', nargs='*', help='atom1 atom2 [min_dist] [max_dist] or atom1 atom2 [max_dist]')
    parser.add_argument("--emin", type=float, default=-5.0, help="Min energy (eV)")
    parser.add_argument("--emax", type=float, default=5.0, help="Max energy (eV)")

    # If no arguments, use old interactive mode
    if len(sys.argv) == 1:
        interactive_mode()
        return

    try:
        args = parser.parse_args()
    except:
        old_command_line_mode()
        return

    plot = args.plot
    atoms_args = args.atoms

    if len(atoms_args) >= 2:
        atom1, atom2 = atoms_args[0], atoms_args[1]

        if len(atoms_args) == 3:
            min_dist = 0.0
            max_dist = float(atoms_args[2])
        elif len(atoms_args) == 4:
            min_dist = float(atoms_args[2])
            max_dist = float(atoms_args[3])
        else:
            min_dist = 0.0
            max_dist = 5.0

        quick_icobi(args.emin, args.emax, atom1, atom2, min_dist, max_dist, plot=plot)
    else:
        interactive_mode(plot)

def old_command_line_mode():
    """Original command-line mode for compatibility"""
    if len(sys.argv) >= 4:
        atom1, atom2 = sys.argv[1], sys.argv[2]

        if len(sys.argv) == 4:
            max_dist = float(sys.argv[3])
            min_dist = 0.0
        elif len(sys.argv) == 5:
            min_dist = float(sys.argv[3])
            max_dist = float(sys.argv[4])
        else:
            print("❌ Usage: python quick_cobi.py [--plot] atom1 atom2 max_dist")
            return

        quick_icobi(atom1, atom2, min_dist, max_dist)

def interactive_mode(plot=False):
    """Interactive user mode"""
    try:
        # Try to auto-detect atom types
        if os.path.exists("POSCAR"):
            structure = Structure.from_file("POSCAR")
            species = sorted(list(set([str(site.specie) for site in structure.sites])))
            print(f"Atoms detected in POSCAR: {', '.join(species)}")
        else:
            species = []

        print(f"\nUsage examples:")
        print(f"  python {sys.argv[0]} --plot Pb N 3.0        # With plotting")
        print(f"  python {sys.argv[0]} Pb N 2.0 3.5           # With min and max distances")
        print(f"  python {sys.argv[0]} -p Fe O 1.5 2.8        # Plotting + distances")

        print(f"\nInteractive mode:")
        atom1 = input("First atom: ").strip()
        atom2 = input("Second atom: ").strip()

        # Ask for distances
        dist_input = input("Distance (max) or min-max (e.g. '3.0' or '2.0-3.5'): ").strip()

        if '-' in dist_input:
            try:
                min_dist, max_dist = map(float, dist_input.split('-'))
            except ValueError:
                print("❌ Invalid format. Use '2.0-3.5' for min-max")
                return
        else:
            try:
                min_dist = 0.0
                max_dist = float(dist_input)
            except ValueError:
                print("❌ Invalid distance")
                return

        if not plot:
            plot_choice = input("Display COBI curves? (y/N): ").strip().lower()
            plot = plot_choice in ['y', 'yes', 'o', 'oui']

        quick_icobi(atom1, atom2, min_dist, max_dist, plot=plot)

    except KeyboardInterrupt:
        print("\n👋 Cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
