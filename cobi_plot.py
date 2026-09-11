#!/usr/bin/env python3
import argparse
import numpy as np
import sys
from matplotlib import pyplot as plt
import palettable
from pymatgen.electronic_structure.cohp import CompleteCohp, Cohp
from pymatgen.electronic_structure.plotter import CohpPlotter
from pymatgen.electronic_structure.core import Spin
import os

def parse_label_string(s):
    """
    Convertit une chaîne comme "12,15,280,500" ou "1-10,12,15-20" en liste de chaînes.
    Les éléments sont séparés par des virgules. Si un élément contient '-', il est interprété
    comme une plage inclusive (ex: "1-5" donne ["1","2","3","4","5"]).
    """
    numbers = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            debut, fin = part.split('-')
            debut = int(debut)
            fin = int(fin)
            if debut <= fin:
                numbers.extend(str(i) for i in range(debut, fin+1))
            else:
                numbers.extend(str(i) for i in range(debut, fin-1, -1))
        else:
            numbers.append(part)
    return numbers

def read_icobilist(filename="ICOBILIST.lobster"):
    """
    Lit le fichier ICOBILIST.lobster et retourne un dictionnaire {label: valeur_ICOBI}
    en ne prenant que la première ligne de chaque label (valeur totale).
    """
    icobi_dict = {}
    seen_labels = set()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                label = parts[0]          # premier champ : numéro de liaison
                value = float(parts[-1])  # dernier champ : valeur ICOBI
            except (ValueError, IndexError):
                continue
            if label not in seen_labels:
                icobi_dict[label] = value
                seen_labels.add(label)
    return icobi_dict

def read_icobilist_full(filename="ICOBILIST.lobster"):
    """
    Lit ICOBILIST.lobster et renvoie {label: {el1,i1,el2,i2,dist,icobi}}
    en ne gardant que la première ligne de chaque label (valeur totale).
    """
    import re
    pat = re.compile(r"^([A-Za-z]{1,2})(\d+)$")
    entries = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            label = parts[0]
            if label in entries:
                continue
            m1, m2 = pat.match(parts[1]), pat.match(parts[2])
            if not (m1 and m2):
                continue
            try:
                dist = float(parts[3])
                icobi = float(parts[-1])
            except ValueError:
                continue
            entries[label] = {"el1": m1.group(1), "i1": int(m1.group(2)),
                              "el2": m2.group(1), "i2": int(m2.group(2)),
                              "dist": dist, "icobi": icobi}
    return entries


def classify_sites(structure, target, neighbour, cutoff):
    """{indice_site_1based: 'n VOISIN'} pour les sites de l'élément target."""
    mapping = {}
    for i, site in enumerate(structure, start=1):
        try:
            el = site.specie.symbol
        except AttributeError:
            el = site.species_string
        if el != target:
            continue
        count = sum(1 for nb in structure.get_neighbors(site, cutoff)
                    if nb.specie.symbol == neighbour)
        mapping[i] = f"{count} {neighbour}"
    return mapping


def build_auto_groups(entries, pairs, site_classes, distmin, distmax):
    """
    Construit les groupes de labels à partir des paires d'éléments demandées,
    éventuellement subdivisées selon la classe du site (H pontant / terminal).
    Renvoie (labels_str, names_list).
    """
    from collections import OrderedDict
    wanted = {tuple(sorted(p.split('-'))): p.strip() for p in pairs}
    groups = OrderedDict()
    rejected = 0

    for label, e in entries.items():
        if distmax is not None and e["dist"] > distmax:
            rejected += 1
            continue
        if distmin is not None and e["dist"] < distmin:
            rejected += 1
            continue
        key = tuple(sorted([e["el1"], e["el2"]]))
        if key not in wanted:
            continue
        name = wanted[key]
        if site_classes:
            suffix = None
            for idx in (e["i1"], e["i2"]):
                if idx in site_classes:
                    suffix = site_classes[idx]
                    break
            if suffix is not None:
                name = f"{name} [{suffix}]"
        groups.setdefault(name, []).append(label)

    if rejected:
        print(f"  {rejected} liaisons écartées par le filtre de distance")
    if not groups:
        raise ValueError("Aucune liaison retenue : vérifie --pairs et --distmax")

    for name in groups:
        dists = [entries[l]["dist"] for l in groups[name]]
        icos = [entries[l]["icobi"] for l in groups[name]]
        print(f"  {name:<22} {len(groups[name]):>4} liaisons   "
              f"d = {min(dists):.3f}-{max(dists):.3f} Å   "
              f"ICOBI moyen = {np.mean(icos):.6f}")

    labels_str = ';'.join(','.join(v) for v in groups.values())
    return labels_str, list(groups.keys())


def save_command(output_filename, command_line):
    """
    Sauvegarde la ligne de commande dans un fichier .command
    """
    cmd_filename = output_filename + '.command'
    with open(cmd_filename, 'w') as f:
        f.write(command_line + '\n')
    print(f"Commande sauvegardée dans {cmd_filename}")

def plot_cobi_from_args(nbonds, labels_str, names_list, fontsize, ymin, ymax,
                        linewidth, output_filename, command_line,
                        eshift=0.0, zero_line="0", sum_groups=None, combine=None,
                        group_colors=None, dashed_groups=None,
                        show_icobi=False, icobi_fmt=".3f"):
    """
    nbonds : nombre de groupes (interactions)
    labels_str : chaîne contenant les groupes séparés par ';'
    names_list : liste des noms des groupes (doit avoir longueur nbonds)
    fontsize : taille de police pour les textes
    ymin, ymax : limites de l'axe y (optionnelles)
    linewidth : épaisseur des courbes
    output_filename : nom du fichier de sortie
    command_line : ligne de commande à sauvegarder
    sum_groups : noms ou indices (1-based) des groupes à sommer au lieu de moyenner
    """
    # Séparation des groupes de labels
    label_groups = labels_str.split(';')
    if len(label_groups) != nbonds:
        raise ValueError(f"Le nombre de groupes dans --labels ({len(label_groups)}) ne correspond pas à --bonds ({nbonds})")

    # Gestion des noms par défaut si non fournis
    if names_list is None:
        names_list = [f"Group {i+1}" for i in range(nbonds)]
    else:
        if len(names_list) != nbonds:
            raise ValueError(f"Le nombre de noms ({len(names_list)}) ne correspond pas à --bonds ({nbonds})")

    # Lecture du fichier ICOBILIST.lobster
    print("Lecture de ICOBILIST.lobster...")
    icobi_dict = read_icobilist("ICOBILIST.lobster")
    all_labels = list(icobi_dict.keys())
    print(f"Nombre total de liaisons dans ICOBILIST : {len(all_labels)}")
    print("Exemples de labels (10 premiers) :", all_labels[:10])
    print("Exemples de valeurs correspondantes :", [icobi_dict[lbl] for lbl in all_labels[:10]])

    # Chargement des données COBICAR
    print("Chargement de COBICAR.lobster et POSCAR...")
    completecohp = CompleteCohp.from_file(fmt="LOBSTER",
                                          filename="COBICAR.lobster",
                                          structure_file="POSCAR")
    cp = CohpPlotter(are_cobis=True)

    # Traitement de chaque groupe
    group_cohps = {}
    group_icobi = {}
    for idx, (group_str, name) in enumerate(zip(label_groups, names_list)):
        # Conversion de la chaîne en liste de chaînes (labels)
        labels = parse_label_string(group_str)
        print(f"\nGroupe '{name}' : {len(labels)} labels fournis : {labels}")

        # Vérification de l'existence des labels dans ICOBILIST
        valid_labels = []
        ico_values = []
        for lbl in labels:
            if lbl in icobi_dict:
                ico_values.append(icobi_dict[lbl])
                valid_labels.append(lbl)
            else:
                print(f"  Attention : label {lbl} introuvable, ignoré")

        if not valid_labels:
            print(f"  Aucun label valide pour ce groupe, il ne sera pas tracé.")
            continue

        print(f"  {len(valid_labels)} labels valides : {valid_labels}")

        # Somme brute ou moyenne pour ce groupe
        summed = False
        if sum_groups:
            keys_lower = {str(k).strip().lower() for k in sum_groups}
            if name.strip().lower() in keys_lower or str(idx + 1) in keys_lower:
                summed = True
        divisor = 1 if summed else len(valid_labels)

        if summed:
            print(f"  Mode SOMME (non moyennée) sur {len(valid_labels)} liaisons")
            print(f"  ICOBI sommé = {np.sum(ico_values):.6f} "
                  f"(moyen = {np.mean(ico_values):.6f})")
        else:
            print(f"  ICOBI moyen = {np.mean(ico_values):.6f}")

        # Ajout de la COBI au plotter
        cohp_obj = completecohp.get_summed_cohp_by_label_list(
            label_list=valid_labels,
            divisor=divisor,
            summed_spin_channels=True)
        cp.add_cohp(name, cohp_obj)
        group_cohps[name] = cohp_obj
        group_icobi[name] = float(np.sum(ico_values) if summed else np.mean(ico_values))

    # Groupes combinés : somme de courbes déjà moyennées
    if combine:
        for definition in combine:
            if '=' not in definition:
                raise ValueError(f"Définition --combine mal formée : '{definition}' "
                                 "(attendu 'Nom=GroupeA+GroupeB')")
            new_name, expr = definition.split('=', 1)
            new_name = new_name.strip()
            parts = [p.strip() for p in expr.split('+') if p.strip()]
            missing = [p for p in parts if p not in group_cohps]
            if missing:
                raise ValueError(f"Groupe(s) inconnu(s) dans --combine : {missing}. "
                                 f"Groupes disponibles : {list(group_cohps)}")

            ref = group_cohps[parts[0]]
            energies = np.asarray(ref.energies, dtype=float)
            spins = set(ref.cohp)
            for p in parts[1:]:
                spins &= set(group_cohps[p].cohp)

            cohp_sum = {s: sum(np.asarray(group_cohps[p].cohp[s], dtype=float)
                               for p in parts) for s in spins}
            icohp_sum = None
            if all(getattr(group_cohps[p], "icohp", None) for p in parts):
                icohp_sum = {s: sum(np.asarray(group_cohps[p].icohp[s], dtype=float)
                                    for p in parts) for s in spins}

            combined = Cohp(ref.efermi, energies, cohp_sum,
                            are_cobis=True, icohp=icohp_sum)
            cp.add_cohp(new_name, combined)
            group_cohps[new_name] = combined
            total_ico = sum(group_icobi[p] for p in parts)
            group_icobi[new_name] = total_ico
            print(f"\nGroupe combiné '{new_name}' = {' + '.join(parts)}")
            print(f"  ICOBI cumulé = {total_ico:.6f} "
                  f"({', '.join(f'{p}: {group_icobi[p]:.6f}' for p in parts)})")

    # Tracé
    if not cp._cohps:
        print("Aucune courbe à tracer.")
        return

    ncolors = max(3, len(cp._cohps))
    ncolors = min(9, ncolors)
    colors = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

    fig, ax = plt.subplots(figsize=(6, 8))
    keys = list(cp._cohps)

    # Collecte des données pour ajustement automatique des limites x
    all_x_vals = []
    if eshift:
        print(f"Décalage de l'échelle d'énergie : E -> E - {eshift:.4f} eV")

    for key in keys:
        energies = np.asarray(cp._cohps[key]["energies"], dtype=float) - eshift
        populations = cp._cohps[key]["COHP"]  # COBI
        for spin in [Spin.up, Spin.down]:
            if spin in populations:
                ydata = populations[spin]
                # Filtrer selon ymin/ymax si spécifiés
                mask = np.ones_like(energies, dtype=bool)
                if ymin is not None:
                    mask &= (energies >= ymin)
                if ymax is not None:
                    mask &= (energies <= ymax)
                if np.any(mask):
                    all_x_vals.extend(ydata[mask])

    # Tracé des courbes
    def _match(key, spec, idx):
        if not spec:
            return None
        keys_lower = {str(k).strip().lower(): v for k, v in spec.items()}
        return (keys_lower.get(str(key).strip().lower())
                or keys_lower.get(str(idx + 1)))

    dashed_set = set()
    if dashed_groups:
        dashed_set = {str(d).strip().lower() for d in dashed_groups}

    for idx, key in enumerate(keys):
        energies = np.asarray(cp._cohps[key]["energies"], dtype=float) - eshift
        populations = cp._cohps[key]["COHP"]
        color = _match(key, group_colors, idx) or colors[idx % ncolors]
        is_dashed = (str(key).strip().lower() in dashed_set
                     or str(idx + 1) in dashed_set)
        ls_up = "--" if is_dashed else "-"
        ls_down = ":" if is_dashed else "--"
        for spin in [Spin.up, Spin.down]:
            if spin in populations:
                y = populations[spin]
                if spin == Spin.up:
                    label = str(key)
                    if show_icobi and key in group_icobi:
                        label = (f"{key} (ICOBI = "
                                 f"{group_icobi[key]:{icobi_fmt}})")
                    plt.plot(y, energies, color=color,
                             linestyle=ls_up, label=label, linewidth=linewidth)
                else:
                    plt.plot(y, energies, color=color,
                             linestyle=ls_down, linewidth=linewidth)

    # Lignes horizontale et verticale (épaisseur fixe 1)
    if str(zero_line).strip().lower() != "none":
        plt.axhline(y=float(zero_line), color="black", linestyle="dashed", linewidth=1.5)
    plt.axvline(x=0, color="black", linestyle="dashed", linewidth=1.5)

    plt.subplots_adjust(top=0.98, bottom=0.10, left=0.12, right=0.96)
    plt.legend(frameon=False, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel("Energy (eV)", fontsize=fontsize)
    plt.xlabel("COBI", fontsize=fontsize)

    # Application des limites d'énergie si fournies
    if ymin is not None or ymax is not None:
        plt.ylim(ymin, ymax)

    # Ajustement automatique des limites x en fonction des données dans la fenêtre y
    if all_x_vals:
        xmin_data = min(all_x_vals)
        xmax_data = max(all_x_vals)
        # Ajouter une marge de 5%
        margin = 0.05 * (xmax_data - xmin_data) if xmax_data != xmin_data else 0.05 * abs(xmin_data) or 0.01
        plt.xlim(xmin_data - margin, xmax_data + margin)
    output_filenamend=output_filename+".png"
    plt.savefig(output_filenamend, format="png", dpi=300)
    plt.show()
    print(f"Figure sauvegardée sous {output_filename}")

    # Sauvegarde de la commande
    save_command(output_filename, command_line)

def main():
    parser = argparse.ArgumentParser(description="Trace les COBI moyennées pour des groupes de labels de liaisons.")
    parser.add_argument('--bonds', type=int, default=None,
                        help="Nombre de groupes (interactions) à tracer. "
                             "Inutile avec --pairs.")
    parser.add_argument('--labels', type=str, default=None,
                        help="Liste des labels pour chaque groupe, séparés par des points-virgules. "
                             "Exemple : '12,15,280,500;1-10,12,15-20'")
    parser.add_argument('--pairs', type=str, default=None,
                        help="Construit les groupes automatiquement à partir de "
                             "ICOBILIST.lobster. Paires d'éléments séparées par ';'. "
                             "Exemple : 'Na-H;Be-H'")
    parser.add_argument('--classify', type=str, default=None,
                        help="Subdivise les groupes selon la classe du site : "
                             "'VOISIN:CUTOFF[:CIBLE]', ex. 'Be:1.8:H' sépare les "
                             "liaisons impliquant un H pontant d'un H terminal")
    parser.add_argument('--distmin', type=float, default=None,
                        help="Distance minimale des liaisons retenues (Å)")
    parser.add_argument('--distmax', type=float, default=None,
                        help="Distance maximale des liaisons retenues (Å). "
                             "Indispensable avec --pairs : cohpGenerator va "
                             "jusqu'à 6 Å et inclut des paires non liées.")
    parser.add_argument('--poscar', type=str, default="POSCAR",
                        help="Structure utilisée par --classify (défaut: POSCAR)")
    parser.add_argument('--names', type=str,
                        help="Noms des groupes, séparés par des points-virgules (optionnel). "
                             "Exemple : 'Pb-Pb;N-N'")
    parser.add_argument('--fontsize', type=int, default=20,
                        help="Taille de police pour les axes et la légende (défaut: 20)")
    parser.add_argument('--ymin', type=float, default=None,
                        help="Limite inférieure de l'axe y (énergie) [optionnel]")
    parser.add_argument('--ymax', type=float, default=None,
                        help="Limite supérieure de l'axe y (énergie) [optionnel]")
    parser.add_argument('--linewidth', type=float, default=3.0,
                        help="Épaisseur des courbes (défaut: 3.0)")
    parser.add_argument('--show-icobi', dest='show_icobi', action='store_true',
                        help="Affiche l'ICOBI de chaque groupe dans la légende "
                             "(moyen par liaison, ou cumulé pour un groupe "
                             "combiné ou sommé)")
    parser.add_argument('--icobi-fmt', dest='icobi_fmt', default='.3f',
                        help="Format des ICOBI en légende (défaut: .3f)")
    parser.add_argument('--colors', type=str, default=None,
                        help="Couleurs par groupe, format 'Nom=couleur' séparés "
                             "par ';'. Accepte noms ou indices 1-based. "
                             "Ex. '--colors \"Total=black\"'")
    parser.add_argument('--dashed', type=str, default=None,
                        help="Groupes tracés en pointillés, séparés par des "
                             "virgules. Ex. '--dashed Total'")
    parser.add_argument('--combine', type=str, default=None,
                        help="Crée une courbe supplémentaire par somme de groupes "
                             "déjà tracés (moyennés). Plusieurs définitions séparées "
                             "par ';'. Exemple : '--combine \"Total=Na-H+Be-H\"'")
    parser.add_argument('--sum', type=str, default=None,
                        help="Groupes à sommer au lieu de moyenner, séparés par des "
                             "virgules. Accepte les noms ou les indices 1-based. "
                             "Exemple : '--sum Total' ou '--sum 3'")
    parser.add_argument('--eshift', type=float, default=0.0,
                        help="Décalage de l'échelle d'énergie en eV (E -> E - eshift). "
                             "Utiliser la valeur imprimée par dos_plot.py pour aligner "
                             "les deux panneaux.")
    parser.add_argument('--zero-line', dest='zero_line', default="0",
                        help="Ordonnée de la ligne pointillée de référence, "
                             "ou 'none' pour la masquer (défaut: 0)")
    parser.add_argument('--output', type=str, default="testcobi",
                        help="Nom du fichier de sortie (défaut: testcobi.png)")
    args = parser.parse_args()

    # Traitement des noms
    names = None
    if args.names:
        names = [n.strip() for n in args.names.split(';')]
        print(names)

    # Reconstruction de la ligne de commande
    command_line = ' '.join(sys.argv)

    # Groupes à sommer plutôt qu'à moyenner
    sum_groups = None
    if args.sum:
        sum_groups = [s.strip() for s in args.sum.split(',') if s.strip()]
        print("Groupes sommés (non moyennés) :", sum_groups)

    # Groupes combinés
    combine = None
    if args.combine:
        combine = [c.strip() for c in args.combine.split(';') if c.strip()]
        print("Groupes combinés :", combine)

    # Construction automatique des groupes depuis ICOBILIST.lobster
    nbonds, labels = args.bonds, args.labels
    if args.pairs:
        if args.labels:
            sys.exit("--pairs et --labels sont exclusifs")
        print("Construction automatique des groupes depuis ICOBILIST.lobster...")
        entries = read_icobilist_full("ICOBILIST.lobster")
        print(f"  {len(entries)} liaisons lues")
        if args.distmax is None:
            print("  ATTENTION : pas de --distmax, toutes les paires jusqu'au "
                  "rayon du cohpGenerator seront incluses")

        site_classes = None
        if args.classify:
            from pymatgen.core import Structure
            bits = args.classify.split(':')
            if len(bits) < 2:
                sys.exit("--classify attend 'VOISIN:CUTOFF[:CIBLE]', ex. 'Be:1.8:H'")
            neighbour, cutoff = bits[0].strip(), float(bits[1])
            target = bits[2].strip() if len(bits) > 2 else "H"
            structure = Structure.from_file(args.poscar)
            site_classes = classify_sites(structure, target, neighbour, cutoff)
            counts = {}
            for cls in site_classes.values():
                counts[cls] = counts.get(cls, 0) + 1
            print(f"  Classification des {target} : " +
                  ", ".join(f"{v} sites à {k}" for k, v in sorted(counts.items())))

        pairs = [p.strip() for p in args.pairs.split(';') if p.strip()]
        labels, auto_names = build_auto_groups(entries, pairs, site_classes,
                                               args.distmin, args.distmax)
        nbonds = len(auto_names)
        if names is None:
            names = auto_names
        elif len(names) != nbonds:
            sys.exit(f"--names donne {len(names)} noms pour {nbonds} groupes "
                     f"construits : {auto_names}")
    elif not (args.bonds and args.labels):
        sys.exit("Il faut fournir soit --pairs, soit --bonds et --labels")

    # Couleurs et styles personnalisés
    group_colors = None
    if args.colors:
        group_colors = {}
        for item in args.colors.split(';'):
            if '=' in item:
                k, v = item.split('=', 1)
                group_colors[k.strip()] = v.strip()
    dashed_groups = None
    if args.dashed:
        dashed_groups = [d.strip() for d in args.dashed.split(',') if d.strip()]

    plot_cobi_from_args(nbonds, labels, names,
                        args.fontsize, args.ymin, args.ymax,
                        args.linewidth, args.output, command_line,
                        eshift=args.eshift, zero_line=args.zero_line,
                        sum_groups=sum_groups, combine=combine,
                        group_colors=group_colors, dashed_groups=dashed_groups,
                        show_icobi=args.show_icobi, icobi_fmt=args.icobi_fmt)

if __name__ == "__main__":
    main()
