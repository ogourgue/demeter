# Demeter

**Biogeomorphic modeling of coastal wetlands** - Couples vegetation dynamics with hydro-morphodynamics to study saltmarsh and mangrove evolution.

## Quick Start

### Installation
```bash
pip install git+https://github.com/ogourgue/demeter.git
```

**Requirements:** Python 3.6+, numpy, scipy, matplotlib, mpi4py (auto-installed)

### Optional: Telemac Coupling

For Telemac integration, install [pputils](https://codeberg.org/pprodano/pputils) manually:
```bash
git clone https://codeberg.org/pprodano/pputils.git
export PYTHONPATH="${PYTHONPATH}:/path/to/pputils"
```

Supported Telemac versions: **v8p2r0, v8p2r1** (more coming soon)

## Learn More

**[Documentation & Examples](https://github.com/yourusername/demeter/wiki)**
- Tutorials and workflows
- Example applications (mangroves, salt marshes)
- Theory and model description
- API reference

## Citation

If you use Demeter in your research:
```bibtex
@article{gourgue2024,
  author = {Gourgue, Olivier and Belliard, Jean-Philippe and Xu, Yiyang and Kleinhans, Maarten G. and Fagherazzi, Sergio and Temmerman, Stijn},
  title = {Dense vegetation hinders sediment transport toward saltmarsh interiors},
  journal = {Limnology and Oceanography Letters},
  volume = {9},
  number = {6},
  pages = {764-775},
  doi = {https://doi.org/10.1002/lol2.10436},
  year = {2024}
}
```

See all [publications using Demeter](https://github.com/yourusername/demeter/wiki/Publications)

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE)

## Contact & Support

Olivier Gourgue - ogourgue@gmail.com

[Start a discussion](https://github.com/yourusername/demeter/discussions) - Questions, ideas, or feedback  
[Report issues](https://github.com/yourusername/demeter/issues) - Bug reports and feature requests
