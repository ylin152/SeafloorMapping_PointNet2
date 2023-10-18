# Anders Knudby, April 2021
# This code converts ICESat-2 .h5 files to csv files

# Imports
import glob
import shutil
import pandas as pd
from pyproj import Transformer
import os
import io
import re
import h5py
import numpy as np
import scipy.interpolate
import math
import argparse


# Functions
def deleteFileIfExists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


def findSurface(input, minElev, maxElev):
    df = input

    hist = np.histogram(df["elev"], int(maxElev - minElev))
    largest_bin = np.argmax(hist[0])

    # Get subset of histogram, +/- 2m around the largest bin
    if largest_bin >= 2 and largest_bin <= len(hist[0]) - 2:
        numbers = hist[0][largest_bin - 2:largest_bin + 3]
        elevations = hist[1][largest_bin - 2:largest_bin + 3]
    elif largest_bin == len(hist[0]) - 1:
        numbers = hist[0][largest_bin - 2:largest_bin + 3]
        elevations = hist[1][largest_bin - 2:largest_bin + 3]
    elif largest_bin == 1:  # This can happen when there is no land in the dataset, and no atmospheric noise
        numbers = hist[0][largest_bin - 1:largest_bin + 3]
        elevations = hist[1][largest_bin - 1:largest_bin + 3]
    elif largest_bin == 1:
        numbers = hist[0][largest_bin:largest_bin + 3]
        elevations = hist[1][largest_bin:largest_bin + 3]
    else:
        print("Weird depth distribution of points, check visually")

    df_subset = df[(df["elev"] > elevations[0]) & (df["elev"] < elevations[min(len(elevations) - 1, 4)])]

    mean = np.mean(df_subset["elev"])
    sd = np.std(df_subset["elev"])

    # set water surface as 5
    df.loc[(df['elev'] > mean - 2 * sd) & (df['elev'] < mean + 2 * sd), 'class'] = 5
    # only keep the points that are lower than average water surface level to decrease data volume
    df = df[df["elev"] < mean]

    # print("Done finding water surface")
    return df


# setting
parser = argparse.ArgumentParser(description='Convert ATL03 to CSV file')
parser.add_argument('--data_dir', type=str, required=True, help='Input directory')
parser.add_argument('--maxElev', type=int, default=10, help='Maximum elevation for filter')
parser.add_argument('--minElev', type=int, default=-50, help='Minimum elevation for filter')
parser.add_argument('--removeLand', action='store_true')
parser.add_argument('--removeIrrelevant', action='store_true')
parser.add_argument('--utm', action='store_true')
parser.add_argument('--interval', default=100000)


def convert(dataDir, utm=True, removeLand=True, removeIrrelevant=True, interval=100000, maxElev=10, minElev=-50):
    filenames = glob.glob(dataDir + "/*.h5")

    output_dir = os.path.join(dataDir, 'csv_data')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for filename in filenames:

        # print(filename)
        print('Start converting H5 to CSV...')

        with h5py.File(filename, mode='r') as fileID:
            # List all groups
            # print("Keys: %s" % fileID.keys())

            # -- allocate python dictionaries for ICESat-2 ATL03 variables and attributes
            IS2_atl03_mds = {}
            IS2_atl03_attrs = {}

            # -- read each input beam within the file
            IS2_atl03_beams = []
            for gtx in [k for k in fileID.keys() if bool(re.match(r'gt\d[lr]', k))]:
                try:
                    fileID[gtx]['geolocation']['reference_photon_lat']
                    fileID[gtx]['heights']['delta_time']
                except KeyError:
                    pass
                else:
                    IS2_atl03_beams.append(gtx)

                # -- for each included beam
            for gtx in IS2_atl03_beams:
                # -- get each HDF5 variable
                IS2_atl03_mds[gtx] = {}
                IS2_atl03_mds[gtx]['heights'] = {}
                IS2_atl03_mds[gtx]['geolocation'] = {}
                # IS2_atl03_mds[gtx]['bckgrd_atlas'] = {}
                IS2_atl03_mds[gtx]['geophys_corr'] = {}
                # -- ICESat-2 Measurement Group
                for key, val in fileID[gtx]['heights'].items():
                    IS2_atl03_mds[gtx]['heights'][key] = val[:]
                # -- ICESat-2 Geolocation Group
                for key, val in fileID[gtx]['geolocation'].items():
                    IS2_atl03_mds[gtx]['geolocation'][key] = val[:]
                # -- ICESat-2 Background Photon Rate Group
                # for key, val in fileID[gtx]['bckgrd_atlas'].items():
                #     IS2_atl03_mds[gtx]['bckgrd_atlas'][key] = val[:]
                # -- ICESat-2 Geophysical Corrections Group: Values for tides (ocean,
                # -- solid earth, pole, load, and equilibrium), inverted barometer (IB)
                # -- effects, and range corrections for tropospheric delays
                for key, val in fileID[gtx]['geophys_corr'].items():
                    IS2_atl03_mds[gtx]['geophys_corr'][key] = val[:]

                # -- Getting attributes of IS2_atl03_mds beam variables
                IS2_atl03_attrs[gtx] = {}
                IS2_atl03_attrs[gtx]['heights'] = {}
                IS2_atl03_attrs[gtx]['geolocation'] = {}
                # IS2_atl03_attrs[gtx]['bckgrd_atlas'] = {}
                IS2_atl03_attrs[gtx]['geophys_corr'] = {}
                # -- Global Group Attributes
                for att_name, att_val in fileID[gtx].attrs.items():
                    IS2_atl03_attrs[gtx][att_name] = att_val
                # -- ICESat-2 Measurement Group
                for key, val in fileID[gtx]['heights'].items():
                    IS2_atl03_attrs[gtx]['heights'][key] = {}
                    for att_name, att_val in val.attrs.items():
                        IS2_atl03_attrs[gtx]['heights'][key][att_name] = att_val
                # -- ICESat-2 Geolocation Group
                for key, val in fileID[gtx]['geolocation'].items():
                    IS2_atl03_attrs[gtx]['geolocation'][key] = {}
                    for att_name, att_val in val.attrs.items():
                        IS2_atl03_attrs[gtx]['geolocation'][key][att_name] = att_val
                # -- ICESat-2 Background Photon Rate Group
                # for key,val in fileID[gtx]['bckgrd_atlas'].items():
                #     IS2_atl03_attrs[gtx]['bckgrd_atlas'][key] = {}
                #     for att_name,att_val in val.attrs.items():
                #         IS2_atl03_attrs[gtx]['bckgrd_atlas'][key][att_name]=att_val
                # -- ICESat-2 Geophysical Corrections Group
                for key, val in fileID[gtx]['geophys_corr'].items():
                    IS2_atl03_attrs[gtx]['geophys_corr'][key] = {}
                    for att_name, att_val in val.attrs.items():
                        IS2_atl03_attrs[gtx]['geophys_corr'][key][att_name] = att_val

            # Ok, now write to csv files
            for gtx in IS2_atl03_beams:

                # Put data in pandas df
                df_data = pd.DataFrame()
                df_data['delta_time'] = IS2_atl03_mds[gtx]['heights']["delta_time"]
                df_data['h_ph'] = IS2_atl03_mds[gtx]['heights']["h_ph"]
                df_data['lat_ph'] = IS2_atl03_mds[gtx]['heights']["lat_ph"]
                df_data['lon_ph'] = IS2_atl03_mds[gtx]['heights']["lon_ph"]

                # This finds the maximum confidence value of any category
                df_data['signal_conf_ph'] = np.amax(IS2_atl03_mds[gtx]['heights']["signal_conf_ph"], axis=1)

                # Put reference information in pandas df

                df_ref = pd.DataFrame()
                df_ref['ref_lat'] = IS2_atl03_mds[gtx]['geolocation']['reference_photon_lat']
                df_ref['ref_lon'] = IS2_atl03_mds[gtx]['geolocation']['reference_photon_lon']
                df_ref['ref_geoid'] = IS2_atl03_mds[gtx]['geophys_corr']['geoid']
                df_ref['ref_tide'] = IS2_atl03_mds[gtx]['geophys_corr']['tide_ocean']
                df_ref['ref_dem'] = IS2_atl03_mds[gtx]['geophys_corr']['dem_h']

                # Remove NAs
                df_ref = df_ref[df_ref['ref_geoid'] < 90]
                df_ref = df_ref[df_ref['ref_geoid'] > -105]

                # Remove data outside reference scope
                df_data = df_data[df_data['lat_ph'] > min(df_ref['ref_lat'])]
                df_data = df_data[df_data['lat_ph'] < max(df_ref['ref_lat'])]

                # Remove data with low confidence - 3 is medium confidence, 4 is high confidence
                df_data = df_data[df_data['signal_conf_ph'] >= 3]

                # Interpolate geoid heights to photon latitudesC
                x = df_ref['ref_lat'].to_numpy()
                y = df_ref['ref_geoid'].to_numpy()
                f = scipy.interpolate.interp1d(x, y)
                x_new = df_data['lat_ph']
                y_new = f(x_new)
                df_data['geoid'] = y_new

                # Interpolate tide heights to photon latitudes
                x = df_ref['ref_lat'].to_numpy()
                y = df_ref['ref_tide'].to_numpy()
                f = scipy.interpolate.interp1d(x, y)
                x_new = df_data['lat_ph']
                y_new = f(x_new)
                df_data['tide'] = y_new

                # Interpolate DEM heights to photon latitudes
                x = df_ref['ref_lat'].to_numpy()
                y = df_ref['ref_dem'].to_numpy()
                f = scipy.interpolate.interp1d(x, y)
                x_new = df_data['lat_ph']
                y_new = f(x_new)
                df_data['dem'] = y_new

                # Not applying tide here, because we want the location relative to the geoid (msl)
                df_data['elev'] = df_data['h_ph'] - df_data['geoid']

                # Remove data outside reasonable boundaries
                df_data = df_data[df_data['elev'] > -12000]
                df_data = df_data[df_data['lat_ph'] < 9000]

                if utm:
                    # Find UTM zone to reproject to:
                    zoneToUse = math.ceil((df_data["lon_ph"].mean() + 180) / 6)
                    if df_data["lat_ph"].mean() > 0:
                        outEPSG = str(326) + str(zoneToUse)
                        zone = str(zoneToUse) + "N"
                    else:
                        outEPSG = str(327) + str(zoneToUse)
                        zone = str(zoneToUse) + "S"

                    transformer = Transformer.from_crs("epsg:4326", "epsg:" + outEPSG)
                    df_data["x"], df_data["y"] = transformer.transform(df_data["lat_ph"].to_numpy(),
                                                                       df_data["lon_ph"].to_numpy())

                # Remove irrelevant photons (deeper than 50m, higher than 20m)
                if removeIrrelevant:
                    df_data = df_data[(df_data["elev"] > minElev) & (df_data["elev"] < maxElev)]

                if removeLand:  # Remove photons for which the DEM is more than 50m above the geoid
                    df_data = df_data[(df_data["dem"] - df_data["geoid"] < 50)]

                # Remove unwanted columns
                df = df_data.drop(['delta_time', 'h_ph', 'geoid', 'dem'], axis=1)

                # Add class column
                df['class'] = np.full((len(df)), 3)

                # Set nodata to 0 for tides (slightly dangerous)
                tides = np.array(df['tide'].values.tolist())
                df['tide'] = np.where(tides > 1000, 0, tides).tolist()

                if utm:
                    df['lat'] = df['lat_ph']  # Keeping lat and lon here for reference
                    df['lon'] = df['lon_ph']
                    df = df.drop(['lat_ph', 'lon_ph'], axis=1)
                    df = df[['x', 'y', 'lon', 'lat', 'elev', 'tide', 'signal_conf_ph',
                             'class']]  # Change the order of the columns
                else:
                    df['lat'] = df['lat_ph']
                    df['lon'] = df['lon_ph']
                    df = df.drop(['lat_ph', 'lon_ph'], axis=1)
                    df = df[
                        ['lon', 'lat', 'elev', 'tide', 'signal_conf_ph', 'class']]  # Change the order of the columns

                # Do normalization so all values are between 0 and 1
                # df = (df - df.min()) / (df.max() - df.min())

                # Round to three decimals for all variables
                df = df.round(decimals=3)

                # Find water surface
                df_segment_all = pd.DataFrame(columns=df.columns)
                num = math.ceil((df['y'].max() - df['y'].min()) / interval)
                y1 = df['y'].min()
                for i in range(num):
                    y2 = y1 + interval
                    df_segment = df[(df['y'] >= y1) & (df['y'] < y2)].copy()
                    if not df_segment.empty:
                        df_segment = findSurface(df_segment, minElev, maxElev)
                    df_segment_all = pd.concat([df_segment_all, df_segment], ignore_index=True)
                    y1 = y2
                df = df_segment_all

                # Write data to csv file
                if utm:
                    outFilename = output_dir + "/" + os.path.basename(filename)[
                                              :-3] + "_" + gtx + "_raw_" + zone + ".csv"  # Reinstate if
                else:
                    outFilename = output_dir + "/" + os.path.basename(filename)[:-3] + "_" + gtx + "_raw" + ".csv"
                df.to_csv(outFilename, index=False)

        # Move the original files to new folder
        # new_filename = os.path.join(dir, os.path.basename(filename))
        # shutil.move(filename, new_filename)

        # print("H5 to CSV done!")

def main(args):
    dataDir = args.data_dir
    utm = args.utm
    removeLand = args.removeLand
    removeIrrelevant = args.removeIrrelevant
    interval = args.interval
    maxElev = args.maxElev
    minElev = args.minElev
    convert(dataDir, utm=utm, removeLand=removeLand, removeIrrelevant=removeIrrelevant, interval=interval, maxElev=maxElev, minElev=minElev)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
