import unittest
import numpy as np
import numpy.testing as nt

from sorts.radar.radar_controls import RadarControls
from sorts import radars
from sorts import StaticPriorityScheduler

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

    def test_array_slicing_nmax(self):
        '''
        Tests time array slicing using max_point argument (max points per scheduler/controller period)
        '''
        radar = radars.eiscat3d
        controls = RadarControls(radar, None, scheduler=None)

        # time slice parameters
        t = np.linspace(0.0, 4.0, 5).astype(np.float64)
        duration = np.repeat([0.5], 5)

        controls.set_time_slices(t, duration, max_points=2)

        t_th        = np.array([[0.0, 1.0], [2.0, 3.0], [4.0]], dtype=object)
        t_slice_th  = np.array([[0.5, 0.5], [0.5, 0.5], [0.5]], dtype=object)

        assert np.size(t_th) == np.size(controls.t)
        assert np.size(t_slice_th) == np.size(controls.t_slice)

        # verification of content of sliced arrays
        for i in range(len(t_th)):
            nt.assert_array_equal(controls.t[i], np.atleast_1d(t_th[i]).astype(np.float64))
            nt.assert_array_equal(controls.t_slice[i], np.atleast_1d(t_slice_th[i]).astype(np.float64))

    def test_array_slicing_scheduler(self):
        '''
        Tests time array slicing using automatic slicing according to scheduler periods
        '''
        radar = radars.eiscat3d
        scheduler = StaticPriorityScheduler(radar, 0, 2.0)
        controls = RadarControls(radar, None, scheduler=scheduler)

        # time slice parameters
        t = np.linspace(0.0, 4.0, 5).astype(np.float64)
        duration = np.repeat([0.5], 5)

        controls.set_time_slices(t, duration)

        t_th        = np.array([[0.0, 1.0], [2.0, 3.0], [4.0]], dtype=object)
        t_slice_th  = np.array([[0.5, 0.5], [0.5, 0.5], [0.5]], dtype=object)

        assert np.size(t_th) == np.size(controls.t)
        assert np.size(t_slice_th) == np.size(controls.t_slice)

        # verification of content of sliced arrays
        for i in range(len(t_th)):
            nt.assert_array_equal(controls.t[i], np.atleast_1d(t_th[i]).astype(np.float64))
            nt.assert_array_equal(controls.t_slice[i], np.atleast_1d(t_slice_th[i]).astype(np.float64))

    def test_set_get_control(self):
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

                    data_retrieved = controls.get_property_control(property_name, station)

                    assert np.size(data_retrieved) == np.size(data_sliced)

                    for period_id in range(n_periods):
                        nt.assert_array_almost_equal(np.atleast_1d(data_retrieved[period_id]).astype(np.float64), data_sliced[period_id])


    def test_property_control(self):
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
                control_data = controls.get_property_control(prop, tx)

                for period_id in range(controls.n_periods):
                    nt.assert_almost_equal(data_splitted[period_id], control_data[period_id])

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
                control_data = controls.get_property_control(prop, rx)

                for period_id in range(controls.n_periods):
                    nt.assert_almost_equal(data_splitted[period_id], control_data[period_id])

            # check if all controls are present and returned 
            controlled_property_list = controls.get_property_control_list(rx)

            counter = len(properties_rx)
            for prop in controlled_property_list:
                if prop in properties_rx:
                    counter -= 1

                assert prop in properties_rx

            assert counter == 0