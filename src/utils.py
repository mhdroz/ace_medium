from typing import Dict


def show_comparison_table(comparison: Dict):
    """
    Display side-by-side comparison in readable format
    """
    print(f"\n{'='*70}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'='*70}")

    print(f"\n{'Empty Playbook':<35} | {'Learned Playbook':<35}")
    print(f"{'-'*35} | {'-'*35}")
    print(
        f"{'Found: ' + str(comparison['empty_playbook']['labs_found']) + ' labs':<35} | {'Found: ' + str(comparison['learned_playbook']['labs_found']) + ' labs':<35}"
    )

    # Show what was found by both
    empty_names = {
        lab["name"].lower()
        for lab in comparison["empty_playbook"]["most_recent"]["most_recent_labs"]
    }
    learned_names = {
        lab["name"].lower()
        for lab in comparison["learned_playbook"]["most_recent"]["most_recent_labs"]
    }
    both_found = empty_names & learned_names

    print(f"\n{'FOUND BY BOTH:':<70}")
    for lab in comparison["empty_playbook"]["most_recent"]["most_recent_labs"]:
        if lab["name"].lower() in both_found:
            print(f"  ✓ {lab['name']} - {lab['value']} {lab['unit']} - {lab['date']}")

    print(f"\n{'ONLY FOUND WITH LEARNED PLAYBOOK:':<70}")
    for lab in comparison["learned_playbook"]["most_recent"]["most_recent_labs"]:
        if lab["name"].lower() not in both_found:

            print(f"  ✓ {lab['name']} - {lab['value']} {lab['unit']} - {lab['date']}")

    print(f"\n{'='*70}")
    print(f"IMPROVEMENT: +{comparison['improvement']} labs caught")
    print(f"{'='*70}")
