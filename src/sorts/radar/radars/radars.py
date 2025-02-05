RADARS = dict()


def list_radars():
    """Returns a dict listing all avalible Radars and their variants"""
    return {key: list(val.keys()) for key, val in RADARS.items()}


def register_radar(name, variant, generator, override_ok=False):
    """Registers a new radar"""
    if name not in RADARS:
        RADARS[name] = {}

    if variant in RADARS[name] and not override_ok:
        raise ValueError(f"{name} with {variant=} already registered")

    RADARS[name][variant] = generator


def radar_generator(name, variant, override_ok=False):
    """Decorator to automatically register the radar beam generator."""

    def registrator_wrapper(generator):
        register_radar(name, variant, generator, override_ok=override_ok)
        return generator

    return registrator_wrapper


def get_radar(name, variant, *args, **kwargs):
    """Get a predefined radar instance from the available library of radars.


    Parameters
    ----------
    name : str
        Name of the radar system
    variant : any
        Variant identifier of the radar system
    *args : required
        Additional required positional arguments by the radar instance.
    **kwargs : optional
        Additional keyword arguments supplied to the radar instance.

    Returns
    -------
    sorts.radar.Radar
        A radar instance.

    """
    if name not in RADARS:
        raise ValueError(
            f'"{name}" radar not found. See available Radars:\n'
            + ", ".join([str(x) for x in RADARS])
        )

    if variant not in RADARS[name]:
        raise ValueError(
            f'"{variant}" variant not found. See available variants for {name}:\n'
            + ", ".join([str(x) for x in RADARS[name]])
        )

    radar_generator = RADARS[name][variant]
    radar = radar_generator(*args, **kwargs)
    return radar
