.. currentmodule:: sorts.radar.system

.. _radar:

================================
Radar (sorts.radar.system.radar)
================================

Description
-----------
``SORTS`` simulates the radar measurement process of the ``radial velocity`` and ``position`` of Near-Earth space Objects (NEOs) such as meteors and space debris. As such, the :class:`Radar<radar.Radar>` class not only defines the architecture of the radar system (i.e. network of :class:`Stations<station.Station>`), but it also implements a set of methods useful for handling low-level functions related to measurement and controlling of :class:`radar stations<station.Station>`). 

.. autosummary::
   :toctree: radar/

   ~radar.Radar


.. _predifined_radar_instances

Predifined Radar systems
------------------------
SORTS implementation includes a set of predifined existing :class:`Radar` instances which can be used to quickly define a measurement configuration. Those instances are located in the ``sorts.radars`` module.

1. Instances
````````````
Thhe current implementation includes the following predifined Radar instances :

.. _radar_eiscat3d:

* sorts.radars.eiscat3d
    EISCAT_3D [1]_ is an international research infrastructure performing radar measurements and incohere scatter technique for studies of the atmosphere and near-Earth space environment above the Fenno-Scandinavian Arctic as well as for support of the solar system and radio astronomy sciences.

    In its final configuration, the EISCAT_3D system will consist of a set multistatic phased-array antenna fields (5 in total) located in Finland, Norway and Sweden.

    The ``eiscat3d`` instance uses an exact gain pattern computation which performs an ideal summation of the individual gain of each antenna of the phased array (which in return has a strong impact on performances).

.. _radar_eiscat3d_interp:

* sorts.radars.eiscat3d_interp
    the ``eiscat3d_interp`` instance reduces computational overhead while computing the antenna gain in a certain direction. By taking advantage of the `far-field approximation` and by assuming the homogeneity of antenna properties within the phased array, the theoretical gain pattern can beinterpolated. By performing a change of coordinates, one is then able to compute the antenna gain in any direction.

.. _radar_eiscat3d_demonstrator:

* sorts.radars.eiscat3d_demonstrator
    The ``eiscat3d_demonstrator`` instance implements the current EISCAT_3D demonstrator located in Kiruna. The antenna gain is computed exactly by summing the gains from all individual antenna in the phased-array.

.. _radar_eiscat3d_demonstrator_interp:

* sorts.radars.eiscat3d_demonstrator_interp
    The ``eiscat3d_demonstrator_interp`` instance implements the current EISCAT_3D demonstrator located in Kiruna. The beam pattern is computed by performing a 2D interpolation of a pre-computed beam pattern.

.. _radar_eiscat_uhf:

* sorts.radars.eiscat_uhf
    The EISCAT UHF system [2]_ is a tristatic radar which stations are located in Finland (Sodankylä), Norway (Tromsø) and Sweden (Kiruna). This system has been used multiple times for the observation of space objects such as Space Debris, but also for incoherent radar measurements of electron density fluctuations (used for example for the characterization of auroras and meteors).

    The UHF system operates in the :math:`930 MHz` band with transmitter peak power :math:`2.0 MW`, :math:`12.5 %` duty cycle and :math:`1 µs – 10 ms` pulse length with frequency and phase modulation capability.

.. _radar_eiscat_esr:

* sorts.radars.eiscat_esr
    ESR is a bistatic antenna system located near Longyearbyen in Spitzbergen. ESR operates in the :math:`500 MHz` band with a transmitter peak power of :math:`1000 kW`, :math:`25 %` duty cycle and :math:`1 µs` – :math:`2 ms` pulse length with frequency and phase modulation capability. 

    Its main purpose being the study of Auroras, it can also be used for the characterization of space objects such as meteors and space debris.

.. _radar_tsdr:

* sorts.radars.tsdr
    The ``Tromsø Space Debris Radar`` (located 69.35° N, 19.13° E)
    #TODO 

.. _radars_tsdr_fence:

* sorts.radars.tsdr_fence
    #TODO 

.. _radars_tsdr_phased:

* sorts.radars.tsdr_phased
    #TODO 

.. _radars_tsdr_phased_fence:

* sorts.radars.tsdr_phased_fence
    #TODO 

.. _radar_mock:

* sorts.radars.mock
    The ``mock`` instance creates a monostatic radar operating in the :math:`100 MHz` band. The station (Rx/Tx) is located at the North pole 
    (90° N, 0° E).

    .. note::
        This instance can be used for testing, but keep in mind that does not model an existing system.



2. Examples
```````````
Importing and using predifined RADAR instances can be achieved as follow :

>>> # import main library capabilities
>>> import sorts                           
>>> 
>>> radar = sorts.radars.eiscat3d           # use predifined EISCAT_3D radar system
>>> radar = sorts.radars.eiscat3d_interp    # use predifined EISCAT_3D radar system with interpolated radar gain pattern
>>> radar = sorts.radars.eiscat_uhf         # use predifined EISCAT UHF radar system

The resulting radar object can then be used as a standard :class:`Radar` object. As a simple example, 
one can get all station indices and types as follow : 

>>> radar = sorts.radars.eiscat3d           # use predifined EISCAT_3D radar system
>>> stations = radar.tx + radar.rx          # group all stations in a single object
>>> inds = []
>>> for id_, station in enumerate(stations):
>>>     inds.append((radar.get_station_id(stations), station.type))    
>>> inds
inds = [(0, "tx"),
        (0, "rx"),
        (1, "rx"),
        (2, "rx")]

3. API 
``````
The :package:`sorts.radar.instances` includes a set of functions and classes used to create the predifined Radar instances.

**Classes**

.. autosummary::
	:toctree: instances/

	~instances.RadarSystemsGetter

**Modules**

.. autosummary::
	:toctree: instances/

	~instances.eiscat_3d   
	~instances.eiscat_esr   
	~instances.eiscat_uhf  
	~instances.mock  
	~instances.tsdr  

4. Reference
````````````
.. [1] EISCAT Scientific Collaboration, "What is EISCAT_3D?", `https://eiscat.se/eiscat3d-information/ <https://eiscat.se/eiscat3d-information/>`_

.. [2] Szasz C., Kero J., Pellinen-Wannberg A., Meisel D.D., Wannberg G., Westman A. (2007). Estimated Visual Magnitudes of the EISCAT UHF Meteors. In: Trigo-Rodríguez, J.M., Rietmeijer, F.J.M., Llorca, J., Janches, D. (eds) Advances in Meteoroid and Meteor Science. Springer, New York, NY`https://doi.org/10.1007/978-0-387-78419-9_52 <https://doi.org/10.1007/978-0-387-78419-9_52>`_
