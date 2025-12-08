clibsorts = Extension(
    name="sorts.clibsorts",
    sources=[
        "src/clibsorts/radar_controller.c",
        "src/clibsorts/static_priority_scheduler.c",
        "src/clibsorts/plotting_controls.c",
        "src/clibsorts/signals.c",
        "src/clibsorts/measurements.c",
        "src/clibsorts/radar.c",
    ],
    include_dirs=[
        "src/clibsorts/",
    ],
)
