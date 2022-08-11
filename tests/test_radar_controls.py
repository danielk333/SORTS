import unittest
import numpy as np
import numpy.testing as nt

from sorts.radar.radar_controls import RadarControls
from sorts import radars
from sorts import StaticPriorityScheduler
from sorts import radar_controller

class TestRadarControls(unittest.TestCase):
    def test_overlap_detection(self):
        radar = radars.eiscat3d
        controls = RadarControls(radar, None, scheduler=None)

        # time slice parameters
        t = np.linspace(0.0, 9.0, 10).astype(np.float64)
        duration = np.repeat([1.0], 10)

        # test if no overlap is detected
        assert controls.check_time_slice_overlap(t, duration) == False

        # test if overlap is detected
        duration[4] = 1.1
        
        assert controls.check_time_slice_overlap(t, duration) == True

    def test_pdirs(self):
        radar = radars.eiscat3d
        
        t_ecef = np.arange(0.0, 1.0, 0.5)
        pdirs_th = np.array([
                [[1, 0, 0],
                [0, 1, 0]],
                [[0, 0, 1],
                [0.16714033, 0.87943669, 0.44570754]],
            ])

        # time slice parameters
        t           = np.linspace(0.0, 1.0, 2).astype(np.float64)
        duration    = np.repeat([0.5], 2)

        class TestController(radar_controller.RadarController):
            def __init__(self, profiler=None, logger=None):
                super().__init__(profiler=None, logger=None)

            def generate_controls(self, t, t_slice, radar):
                controls = RadarControls(radar, self, scheduler=None)
                controls.set_time_slices(t, duration, max_points=2)

                r = np.array([[1510.3, 875.7], [89741.2, 150.2]])*1e3
                t_ecef = np.arange(0.0, 1.0, 0.5)

                args = (r,t_ecef)
                controls.pdirs = controls.set_pdirs(args)

                return controls

            def compute_pointing_direction(self, controls, period_id, args):
                r, t_ecef = args

                pdirs = dict()

                # compute pointing directions
                tx_ecef = np.array([tx.ecef for tx in radar.tx])
                rx_ecef = np.array([rx.ecef for rx in radar.rx])

                ecef_points = (tx_ecef[None, 0] + pdirs_th[period_id]*r[period_id][:, None]).T

                pdirs_tx = ecef_points[None, None, :, :] - tx_ecef[:, None, :, None]
                pdirs["tx"] = pdirs_tx/np.linalg.norm(pdirs_tx, axis=2)[:, :, None, :]

                pdirs_rx = ecef_points[None, None, :, :] - rx_ecef[:, None, :, None]
                pdirs["rx"] = pdirs_rx/np.linalg.norm(pdirs_rx, axis=2)[:, :, None, :]

                pdirs["t"] = t_ecef[period_id]

                return pdirs
        
        # test setting and recovering
        controller = TestController()
        controls = controller.generate_controls(t, duration, radar)

        r = np.array([[1510.3, 875.7], [89741.2, 150.2]])*1e3
        t_ecef = np.arange(0.0, 1.0, 0.5)
        args = (r,t_ecef)

        for period_id in range(len(controls.t)):
            pdirs_ref = controller.compute_pointing_direction(controls, period_id, args)
            pdirs = controls.get_pdirs(period_id)

            nt.assert_almost_equal(pdirs["t"], pdirs_ref["t"])
            nt.assert_almost_equal(pdirs["tx"], pdirs_ref["tx"])
            nt.assert_almost_equal(pdirs["rx"], pdirs_ref["rx"])


    def test_array_slicing_nmax(self):
        '''
        Tests time array slicing using max_point argument (max points per scheduler/controller period)
        '''
        radar = radars.eiscat3d
        
        # time slice parameters
        t = np.linspace(0.0, 4.0, 5).astype(np.float64)
        duration = np.repeat([0.5], 5)
        priority = 1

        controls = RadarControls(radar, None, scheduler=None, priority=priority)
        controls.set_time_slices(t, duration, max_points=2)

        t_th        = np.array([[0.0, 1.0], [2.0, 3.0], [4.0]], dtype=object)
        t_slice_th  = np.array([[0.5, 0.5], [0.5, 0.5], [0.5]], dtype=object)
        priority_th = np.array([[1, 1], [1, 1], [1]], dtype=object)

        assert np.size(t_th) == np.size(controls.t)
        assert np.size(t_slice_th) == np.size(controls.t_slice)
        assert np.size(priority_th) == np.size(controls.priority)

        # verification of content of sliced arrays
        for i in range(len(t_th)):
            nt.assert_array_equal(controls.t[i], np.atleast_1d(t_th[i]).astype(np.float64))
            nt.assert_array_equal(controls.t_slice[i], np.atleast_1d(t_slice_th[i]).astype(np.float64))
            nt.assert_array_equal(controls.priority[i], np.atleast_1d(priority_th[i]).astype(np.int64))


    def test_array_slicing_scheduler(self):
        '''
        Tests time array slicing using automatic slicing according to scheduler periods
        '''
        radar = radars.eiscat3d
        scheduler = StaticPriorityScheduler(radar, 0, 2.0)
        priority = 1

        t0_scheduler = 15.0

        # time slice parameters
        t = np.linspace(t0_scheduler, t0_scheduler+4.0, 5).astype(np.float64)
        duration = np.repeat([0.5], 5)

        controls = RadarControls(radar, None, scheduler=scheduler, priority=priority)
        controls.set_time_slices(t, duration)

        spliting_indices = np.array([1, 3])
        t_th        = np.hsplit(t, spliting_indices)
        t_slice_th  = np.hsplit(np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=object), spliting_indices)
        priority_th = np.hsplit(np.array([1, 1, 1, 1, 1], dtype=object), spliting_indices)

        assert np.size(t_th) == np.size(controls.t)
        assert np.size(t_slice_th) == np.size(controls.t_slice)
        assert np.size(priority_th) == np.size(controls.priority)

        # verification of content of sliced arrays
        for i in range(len(t_th)):
            nt.assert_array_equal(controls.t[i], np.atleast_1d(t_th[i]).astype(np.float64))
            nt.assert_array_equal(controls.t_slice[i], np.atleast_1d(t_slice_th[i]).astype(np.float64))
            nt.assert_array_equal(controls.priority[i], np.atleast_1d(priority_th[i]).astype(np.int64))


    def test_property_controls(self):
        '''
        Tests the creation and reading of all available radar controls (properties)
        '''
        radar = radars.eiscat3d

        scheduler = StaticPriorityScheduler(radar, 0, 2.0)
        controls = RadarControls(radar, None, scheduler=scheduler)

        data            = np.array([0, 200, 0.0, 5.184, -14.565]).astype(np.float64)
        data_sliced     = np.array([[0.0, 200], [0.0, 5.184], [-14.565]], dtype=object)

        t               = np.linspace(0, 4.0, 5)
        duration        = np.ones(5)*0.5

        controls.set_time_slices(t, duration)
        n_periods = len(controls.t)

        for station_type in ("tx", "rx"):
            stations = getattr(radar, station_type)

            for sid, station in enumerate(stations):
                for property_name in station.PROPERTIES:
                    controls.add_property_control(property_name, station, data)

        for station_type in ("tx", "rx"):
            stations = getattr(radar, station_type)

            for sid, station in enumerate(stations):
                for property_name in station.PROPERTIES:
                    for period_id in range(n_periods):
                        data_retrieved = controls.get_property_control(property_name, station, period_id)
                        print(data_retrieved)
                        nt.assert_array_almost_equal(np.atleast_1d(data_retrieved).astype(np.float64), data_sliced[period_id])


    def test_specific_tx_rx_property_controls(self):
        '''
        Tests the creation and reading of specific radar controls (properties)
        '''
        radar = radars.eiscat3d

        scheduler = StaticPriorityScheduler(radar, 0, 2.0)
        controls = RadarControls(radar, None, scheduler=scheduler)

        data            = np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1]).astype(np.float64)
        data_splitted   = np.array([[1, 1], [1, 1], [0, 0], [0, 1], [0, 1]], dtype=object)

        properties_tx   = ["wavelength", "ipp", "pulse_length"]
        properties_rx   = ["wavelength"]

        t               = np.linspace(0, 9.0, 10)
        duration        = np.ones(10)*0.5
        controls.set_time_slices(t, duration)


        # test TX control properties
        for txi, tx in enumerate(radar.tx):
            for prop in properties_tx:
                controls.add_property_control(prop, tx, data)

        for txi, tx in enumerate(radar.tx):
            for prop in properties_tx:
                for period_id in range(controls.n_periods):
                    control_data = controls.get_property_control(prop, tx, period_id)
                    nt.assert_almost_equal(data_splitted[period_id], control_data)

            # check if all controls are present and returned 
            controlled_property_list = controls.get_property_control_list(tx)

            counter = len(properties_tx)
            for prop in controlled_property_list:
                if prop in properties_tx:
                    counter -= 1

                assert prop in properties_tx
            assert counter == 0

        # test RX control properties
        for rxi, rx in enumerate(radar.rx):
            for prop in properties_rx:
                controls.add_property_control(prop, rx, data)

        for rxi, rx in enumerate(radar.rx):
            for prop in properties_rx:
                for period_id in range(controls.n_periods):
                    control_data = controls.get_property_control(prop, rx, period_id)
                    nt.assert_almost_equal(data_splitted[period_id], control_data)

            # check if all controls are present and returned 
            controlled_property_list = controls.get_property_control_list(rx)

            counter = len(properties_rx)
            for prop in controlled_property_list:
                if prop in properties_rx:
                    counter -= 1

                assert prop in properties_rx

            assert counter == 0


    def test_copy(self):
        """
        Tests deepcopy of radar controls
        """
        radar = radars.eiscat3d

        scheduler = StaticPriorityScheduler(radar, 0, 2.0)
        controls_ref = RadarControls(radar, None, scheduler=scheduler)

        data            = np.array([1, 1, 1]).astype(np.float64)
        data_sliced     = np.array([[1.0, 1.0], [1.0]], dtype=object)
        pdirs           = dict()

        pdirs["tx"]     = np.array([[[8.2, 5.1, 1.4], [1.1, 1.9, 9.8], [10.1, 15.9, 0.1]]])
        pdirs["rx"]     = np.array([[[8.2, 5.1, 1.4], [1.1, 1.9, 9.8], [10.1, 15.9, 0.1]], [[8.2, 5.1, 1.4], [1.1, 1.9, 9.8], [10.1, 15.9, 0.1]], [[8.2, 5.1, 1.4], [1.1, 1.9, 9.8], [10.1, 15.9, 0.1]]])
        pdirs["t"]      = np.array([0.0, 1.0, 2.0])

        properties_tx   = ["wavelength", "ipp", "pulse_length"]
        properties_rx   = ["wavelength"]

        t               = np.linspace(0, 2.0, 3)
        duration        = np.ones(3)*0.5

        controls_ref.set_time_slices(t, duration)
        controls_ref.pdirs = pdirs

        # set TX control properties
        for txi, tx in enumerate(radar.tx):
            for prop in properties_tx:
                controls_ref.add_property_control(prop, tx, data)

        # set RX control properties
        for rxi, rx in enumerate(radar.rx):
            for prop in properties_rx:
                controls_ref.add_property_control(prop, rx, data)

        controls_cpy = controls_ref.copy()

        nt.assert_array_equal(controls_ref.pdirs["tx"], controls_cpy.pdirs["tx"])
        nt.assert_array_equal(controls_ref.pdirs["rx"], controls_cpy.pdirs["rx"])
        nt.assert_array_equal(controls_ref.pdirs["t"], controls_cpy.pdirs["t"])

        assert controls_ref.n_periods == controls_cpy.n_periods
        assert controls_ref.n_control_points == controls_cpy.n_control_points

        for station in radar.rx + radar.tx:
            station_id = radar.get_station_id(station)
            station_type = station.type

            for period_id in range(len(controls_ref.property_controls)):
                for name in controls_ref.property_controls[period_id][station_type].keys():
                    nt.assert_array_equal(controls_ref.property_controls[period_id][station_type][name][station_id], controls_cpy.property_controls[period_id][station_type][name][station_id])
                    nt.assert_array_equal(np.asfarray(data_sliced[period_id]), controls_cpy.property_controls[period_id][station_type][name][station_id])


    def test_get_control_period_id(self):
        """
        Tests computations of control periods
        """
        radar = radars.eiscat3d

        scheduler = StaticPriorityScheduler(radar, 0, 2.0)
        controls_ref = RadarControls(radar, None, scheduler=scheduler)

        controller_t0 = 15.0

        t               = np.linspace(controller_t0, controller_t0 + 9.0, 10) # 15 to 24 step=1
        duration        = np.ones(10)*0.5

        controls_ref.set_time_slices(t, duration)
        t_ref = np.array([[15.0], [16.0, 17.0], [18.0, 19.0], [20.0, 21.0], [22.0, 23.0], [24.0]], dtype=object)
        # control period id 0       1               2           3               4           5
        # scheduler period  7       8               9           10              11          12

        for period_id in range(len(controls_ref.t)):
            nt.assert_array_equal(t[period_id], [])

        # test outside of range
        assert controls_ref.get_control_period_id(5)    == -1 # 10s : t < t0
        assert controls_ref.get_control_period_id(14)   == -1 # 28s : t > t0 + control_duration

        # test inside of range
        assert controls_ref.get_control_period_id(7)    == 0 # 16s
        assert controls_ref.get_control_period_id(10)   == 3 # 20s
        assert controls_ref.get_control_period_id(12)   == 5 # 26s