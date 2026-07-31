#!/usr/bin/env python3
"""Manifest-driven lifecycle entrypoint for the Galaxea A1 real HAL."""

from openral_hal.lifecycle import make_lifecycle_main_from_manifest

main = make_lifecycle_main_from_manifest(node_name="openral_hal_galaxea_a1")

if __name__ == "__main__":
    main()
