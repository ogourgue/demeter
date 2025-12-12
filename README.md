# Demeter

**Biogeomorphic modeling of coastal wetlands:** Couples vegetation dynamics with hydro-morphodynamics to study saltmarsh and mangrove evolution.

[![LOL](./assets/Snapshot_20.png)](https://youtu.be/oSzcoR9WBtI)
*Click the image to watch 200 years of coastal landscape evolution with Demeter.*

## Installation
```bash
pip install git+https://github.com/ogourgue/demeter.git
```
- **Requirements:** Python 3.6+, numpy, scipy, matplotlib, mpi4py (auto-installed)
- **For Telemac integration** (optional):
  - Install [Telemac](http://wiki.opentelemac.org/doku.php?id=installation_on_linux) (supported versions: v8p2r0, v8p2r1 - more coming soon)
  - Install [pputils](https://codeberg.org/pprodano/pputils):
  ```bash
  git clone https://codeberg.org/pprodano/pputils.git
  export PYTHONPATH="${PYTHONPATH}:/path/to/pputils"
  ```

## Citation

If you use Demeter in your research, please cite:

Gourgue, O., Belliard, J.P., Xu, Y.Y., Kleinhans, M.G., Fagherazzi, S., Temmerman, S. (2024), Dense vegetation hinders sediment transport toward saltmarsh interiors, *Limnology and Oceanography Letters*, 9(6), 764-775 ([https://doi.org/10.1002/lol2.10436](https://doi.org/10.1002/lol2.10436))

## Contributors

**Olivier Gourgue** — Lead developer and maintainer  
**Jim van Belzen** — Original architect of the cellular automaton approach  
**Christian Schwarz** — Early collaborator who helped shape the foundational concepts  
**Stijn Temmerman** — Supervisor across all development phases  
**Johan van de Koppel** — Supervisor of the initial project  
**Sergio Fagherazzi** — Co-supervised the Boston postdoc phase where version 2 was developed

Demeter grew from saltmarsh to code through many conversations, debugging sessions, and "wait, what if we tried..." moments. Thanks to everyone who contributed ideas, caught bugs, or asked good questions.

## Support

[Start a discussion](https://github.com/ogourgue/demeter/discussions) - Questions, ideas, or feedback  
[Report issues](https://github.com/ogourgue/demeter/issues) - Bug reports and feature requests  
[Check the wiki](https://github.com/ogourgue/demeter/issues) - Tutorials, references
