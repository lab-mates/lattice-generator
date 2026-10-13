# PyInstaller entry for latticegen.exe (console):
#   latticegen selftest [report.txt]     check every feature of this installation
#   latticegen build octet --method struts ...   the full latticegen command line
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    import desktop.paths  # noqa: F401  (sys.path in a source checkout)
    if sys.argv[1:2] == ["selftest"]:
        from desktop.selftest import run
        sys.exit(run(sys.argv[2] if len(sys.argv) > 2 else None))
    from latticegen.cli import main
    main(sys.argv[1:])
