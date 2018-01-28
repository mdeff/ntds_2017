#! /usr/bin/env python3
import json
from tqdm import tqdm

# SATCAT information on fields

operational_status_dict = {}
operational_status_dict["+"] = "Operational"
operational_status_dict["-"] = "Nonoperational"
operational_status_dict["P"] = "Partially Operational"
operational_status_dict["B"] = "Backup/Standby"
operational_status_dict["S"] = "Spare"
operational_status_dict["X"] = "Extended Mission"
operational_status_dict["D"] = "Decayed"
operational_status_dict["?"] = "Unknown"
operational_status_dict[" "] = "Unknown"

# We want to be able to extract the launchsites full name
# List taken from https://celestrak.com/satcat/launchsites.asp
launch_site_full_name = {}
launch_site_full_name["AFETR"] = "Air Force Eastern Test Range, Florida, USA"
launch_site_full_name["AFWTR"] = "Air Force Western Test Range, California, USA"
launch_site_full_name["CAS"] = "Canaries Airspace"
launch_site_full_name["DLS"] = "Dombarovskiy Launch Site, Russia"
launch_site_full_name["ERAS"] = "Eastern Range Airspace"
launch_site_full_name["FRGUI"] = "Europe's Spaceport, Kourou, French Guiana"
launch_site_full_name["HGSTR"] = "Hammaguira Space Track Range, Algeria"
launch_site_full_name["JSC"] = "Jiuquan Space Center, PRC"
launch_site_full_name["KODAK"] = "Kodiak Launch Complex, Alaska, USA"
launch_site_full_name["KSCUT"] = "Uchinoura Space Center (Fomerly Kagoshima Space Center—University of Tokyo, Japan)"
launch_site_full_name["KWAJ"] = "US Army Kwajalein Atoll (USAKA)"
launch_site_full_name["KYMSC"] = "Kapustin Yar Missile and Space Complex, Russia"
launch_site_full_name["NSC"] = "Naro Space Complex, Republic of Korea"
launch_site_full_name["PLMSC"] = "Plesetsk Missile and Space Complex, Russia"
launch_site_full_name["SEAL"] = "Sea Launch Platform (mobile)"
launch_site_full_name["SEMLS"] = "Semnan Satellite Launch Site, Iran"
launch_site_full_name["SNMLP"] = "San Marco Launch Platform, Indian Ocean (Kenya)"
launch_site_full_name["SRILR"] = "Satish Dhawan Space Centre, India (Formerly Sriharikota Launching Range)"
launch_site_full_name["SUBL"] = "Submarine Launch Platform (mobile), Russia"
launch_site_full_name["SVOBO"] = "Svobodnyy Launch Complex, Russia"
launch_site_full_name["TAISC"] = "Taiyuan Space Center, PRC"
launch_site_full_name["TANSC"] = "Tanegashima Space Center, Japan"
launch_site_full_name["TYMSC"] = "Tyuratam Missile and Space Center, Kazakhstan (Also known as Baikonur Cosmodrome)"
launch_site_full_name["VOSTO"] = "Vostochny Cosmodrome, Russia"
launch_site_full_name["WLPIS"] = "Wallops Island, Virginia, USA"
launch_site_full_name["WOMRA"] = "Woomera, Australia"
launch_site_full_name["WRAS"] = "Western Range Airspace"
launch_site_full_name["WSC"] = "Wenchang Satellite Launch Center, PRC"
launch_site_full_name["XICLF"] = "Xichang Launch Facility, PRC"
launch_site_full_name["YAVNE"] = "Yavne Launch Facility, Israel"
launch_site_full_name["YUN"] = "Yunsong Launch Site (Sohae Satellite Launching Station), Democratic People's Republic of Korea"

# Get full name of sources
# List taken from: https://celestrak.com/satcat/sources.asp
# Note: they are not all countries, some are groups of countries
#       or independent associatons/companies
source_full_name_dict = {}
source_full_name_dict["AB"] = "Arab Satellite Communications Organization"
source_full_name_dict["ABS"] = "Asia Broadcast Satellite"
source_full_name_dict["AC"] = "Asia Satellite Telecommunications Company (ASIASAT)"
source_full_name_dict["ALG"] = "Algeria"
source_full_name_dict["ANG"] = "Angola"
source_full_name_dict["ARGN"] = "Argentina"
source_full_name_dict["ASRA"] = "Austria"
source_full_name_dict["AUS"] = "Australia"
source_full_name_dict["AZER"] = "Azerbaijan"
source_full_name_dict["BEL"] = "Belgium"
source_full_name_dict["BELA"] = "Belarus"
source_full_name_dict["BERM"] = "Bermuda"
source_full_name_dict["BOL"] = "Bolivia"
source_full_name_dict["BRAZ"] = "Brazil"
source_full_name_dict["BGD"] = "Peoples Republic of Bangladesh"
source_full_name_dict["CA"] = "Canada"
source_full_name_dict["CHBZ"] = "China/Brazil"
source_full_name_dict["CHLE"] = "Chile"
source_full_name_dict["CIS"] = "Commonwealth of Independent States (former USSR)"
source_full_name_dict["COL"] = "Colombia"
source_full_name_dict["CZCH"] = "Czech Republic (former Czechoslovakia)"
source_full_name_dict["DEN"] = "Denmark"
source_full_name_dict["ECU"] = "Ecuador"
source_full_name_dict["EGYP"] = "Egypt"
source_full_name_dict["ESA"] = "European Space Agency"
source_full_name_dict["ESRO"] = "European Space Research Organization"
source_full_name_dict["EST"] = "Estonia"
source_full_name_dict["EUME"] = "European Organization for the Exploitation of Meteorological Satellites (EUMETSAT)"
source_full_name_dict["EUTE"] = "European Telecommunications Satellite Organization (EUTELSAT)"
source_full_name_dict["FGER"] = "France/Germany"
source_full_name_dict["FIN"] = "Finland"
source_full_name_dict["FR"] = "France"
source_full_name_dict["FRIT"] = "France/Italy"
source_full_name_dict["GER"] = "Germany"
source_full_name_dict["GHA"] = "Republic of Ghana"
source_full_name_dict["GLOB"] = "Globalstar"
source_full_name_dict["GREC"] = "Greece"
source_full_name_dict["HUN"] = "Hungary"
source_full_name_dict["IM"] = "International Mobile Satellite Organization (INMARSAT)"
source_full_name_dict["IND"] = "India"
source_full_name_dict["INDO"] = "Indonesia"
source_full_name_dict["IRAN"] = "Iran"
source_full_name_dict["IRAQ"] = "Iraq"
source_full_name_dict["IRID"] = "Iridium"
source_full_name_dict["ISRA"] = "Israel"
source_full_name_dict["ISRO"] = "Indian Space Research Organisation"
source_full_name_dict["ISS"] = "International Space Station"
source_full_name_dict["IT"] = "Italy"
source_full_name_dict["ITSO"] = "International Telecommunications Satellite Organization (INTELSAT)"
source_full_name_dict["JPN"] = "Japan"
source_full_name_dict["KAZ"] = "Kazakhstan"
source_full_name_dict["LAOS"] = "Laos"
source_full_name_dict["LTU"] = "Lithuania"
source_full_name_dict["LUXE"] = "Luxembourg"
source_full_name_dict["MALA"] = "Malaysia"
source_full_name_dict["MEX"] = "Mexico"
source_full_name_dict["MNG"] = "Mongolia"
source_full_name_dict["MA"] = "Morocco"
source_full_name_dict["NATO"] = "North Atlantic Treaty Organization"
source_full_name_dict["NETH"] = "Netherlands"
source_full_name_dict["NICO"] = "New ICO"
source_full_name_dict["NIG"] = "Nigeria"
source_full_name_dict["NKOR"] = "Democratic People's Republic of Korea"
source_full_name_dict["NOR"] = "Norway"
source_full_name_dict["O3B"] = "O3b Networks"
source_full_name_dict["ORB"] = "ORBCOMM"
source_full_name_dict["PAKI"] = "Pakistan"
source_full_name_dict["PERU"] = "Peru"
source_full_name_dict["POL"] = "Poland"
source_full_name_dict["POR"] = "Portugal"
source_full_name_dict["PRC"] = "People's Republic of China"
source_full_name_dict["PRES"] = "People's Republic of China/European Space Agency"
source_full_name_dict["RASC"] = "RascomStar-QAF"
source_full_name_dict["ROC"] = "Taiwan (Republic of China)"
source_full_name_dict["ROM"] = "Romania"
source_full_name_dict["RP"] = "Philippines (Republic of the Philippines)"
source_full_name_dict["SAFR"] = "South Africa"
source_full_name_dict["SAUD"] = "Saudi Arabia"
source_full_name_dict["SEAL"] = "Sea Launch"
source_full_name_dict["SES"] = "SES"
source_full_name_dict["SING"] = "Singapore"
source_full_name_dict["SKOR"] = "Republic of Korea"
source_full_name_dict["SPN"] = "Spain"
source_full_name_dict["STCT"] = "Singapore/Taiwan"
source_full_name_dict["SWED"] = "Sweden"
source_full_name_dict["SWTZ"] = "Switzerland"
source_full_name_dict["THAI"] = "Thailand"
source_full_name_dict["TMMC"] = "Turkmenistan/Monaco"
source_full_name_dict["TURK"] = "Turkey"
source_full_name_dict["UAE"] = "United Arab Emirates"
source_full_name_dict["UK"] = "United Kingdom"
source_full_name_dict["UKR"] = "Ukraine"
source_full_name_dict["URY"] = "Uruguay"
source_full_name_dict["US"] = "United States"
source_full_name_dict["USBZ"] = "United States/Brazil"
source_full_name_dict["VENZ"] = "Venezuela"
source_full_name_dict["VTNM"] = "Vietnam"

# SATCAT data
def extract_dict(line):
    """This function extract the informations from satcat dataset
       The information is given by field position in a line.
       Because there's no standardized separator, we need to do an analysis by position of each lines.
    """

    sat_data_dict = {}
    try:
        sat_data_dict["launch_year"] = int(line[0:4])
        sat_data_dict["launch_nbr"] = int(line[5:8])
        sat_data_dict["launch_piece"] = line[8:11]
        sat_data_dict["NORAD"] = int(line[13:18])
        sat_data_dict["multiple_name_flag"] =  True if line[19] == "M" else False
        sat_data_dict["payload_flag"] = True if line[20] == "*" else False
        sat_data_dict["operational_status"] = operational_status_dict[line[21]]
        sat_data_dict["name"] = line[23:47].strip()
        sat_data_dict["source"] = line[49:54].strip()
        sat_data_dict["launch_date"] = line[56:66].strip()
        sat_data_dict["launch_site"] = line[68:73].strip()
        sat_data_dict["decay_date"] = line[75:85].strip()
        sat_data_dict["orbital_period"] = float(line[87:94])
        sat_data_dict["inclination"] = float(line[96:101])
        sat_data_dict["apogee"] = float(line[103:109])
        sat_data_dict["perigee"] = float(line[111:117])
        sat_data_dict["radar_cross_section"] = float(line[119:127]) if line[119:127].strip() != "N/A" else None
        sat_data_dict["orbital_status"] = "EA" if line[129:132].strip() == '' else line[129:132].strip()
    # If there's an error, incomplete line for example, we thrash it
    except (IndexError, ValueError) as e:
        return None
    return sat_data_dict

sat_data_array = []
with open('Dataset/satcat.txt') as f:
    bad_data_count = 0
    for line in tqdm(f):
        local_sat_data_dict = extract_dict(line)
        if local_sat_data_dict:
            sat_data_array.append(local_sat_data_dict)
        else:
            bad_data_count +=1

print("Bad data: {}\nGood data: {}".format(bad_data_count, len(sat_data_array)))



satcat_info_dict = {}
satcat_info_dict["sat_data"] = sat_data_array
satcat_info_dict["operational_status"] = operational_status_dict
satcat_info_dict["launch_site"] = launch_site_full_name
satcat_info_dict["source"] = source_full_name_dict


satcat_info_json = json.dumps(satcat_info_dict, indent=4)
with open("./Dataset/satcat_info.json", 'w') as f:
	print(satcat_info_json, file=f)
