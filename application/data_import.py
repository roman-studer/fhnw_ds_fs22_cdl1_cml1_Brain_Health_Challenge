import glob

import pandas as pd


def import_tabular_data(data_folder="../data/tabular_data/"):
    # TODO: NEUROBAT
    common_columns = ['Phase', 'RID', 'VISCODE']

    dx = pd.read_csv(data_folder + 'clean_DXSUM_ADNIALL.csv')[common_columns + ["DX", "EXAMDATE"]]
    dx['DX'] = dx['DX'].map({1: "NC", 2: 'MCI', 3: 'AD'})
    dx = dx.merge(pd.read_csv(data_folder + 'VISITS.csv'),
                  left_on=['Phase', 'VISCODE'], right_on=['Phase', 'VISCODE'], how='inner')

    moca = pd.read_csv(data_folder + 'clean_MOCA.csv')[common_columns + ["MOCASCORE"]]
    dx = dx.merge(moca, left_on=common_columns, right_on=common_columns, how='left')

    gdscale = pd.read_csv(data_folder + 'clean_GDSCALE.csv')[common_columns + ["GDTOTAL"]]
    dx = dx.merge(gdscale, left_on=common_columns, right_on=common_columns, how='left')

    mmse = pd.read_csv(data_folder + 'clean_MMSE.csv')[common_columns + ["MMSCORE"]]
    dx = dx.merge(mmse, left_on=common_columns, right_on=common_columns, how='left')

    cdr = pd.read_csv(data_folder + 'clean_CDR.csv')[common_columns + ["total_score"]]\
        .rename(columns={'total_score': 'CDRTOTAL'})
    patient_scores = dx.merge(cdr, left_on=common_columns, right_on=common_columns, how='left')

    mean_patient_scores = patient_scores.groupby(['Phase', 'VISCODE', 'DX']).agg(
        {'VISORDER': 'first', 'MOCASCORE': 'mean', 'GDTOTAL': 'mean', 'MMSCORE': 'mean',
         'CDRTOTAL': 'mean', }).reset_index()

    return mean_patient_scores, patient_scores


def import_images(data_folder="../data/raw/"):
    img = pd.read_csv(data_folder + 'images/ADNI1_Complete_1Yr_1.5T_1_20_2022.csv')
    image_files = glob.glob(data_folder + 'images/' + '**/*.nii', recursive=True)

    # To access image filenames by their ID: key=image-id, value=filename.
    image_map = {}
    for filename in image_files:
        start_index = filename.rindex('_I') + 1
        end_index = filename.rindex('.nii')
        image_id = filename[start_index:end_index]

        image_map[image_id] = filename

    img['filename'] = img['Image Data ID'].apply(lambda x: image_map.get(x))
    img['RID'] = img['Subject'].apply(lambda s: s[s.rindex('_') + 1:]).astype(int)
    img_columns = ['RID', 'Visit', 'Group', 'filename', 'Age', 'Sex', 'Acq Date']

    def process_date(old_date):
        month, day, year = old_date.split("/")
        return year + '-' + month + '-' + day
    img['Acq Date'] = img['Acq Date'].apply(process_date)

    img = img[img_columns].sort_values(by=['RID', 'Visit']).reset_index(drop=True).dropna()
    return img


def get_demographic_data(data_folder="../data/tabular_data/"):
    demographic_data_1 = pd.read_csv(data_folder + 'clean_ADNIMERGE.csv')
    demographic_data_2 = pd.read_csv(data_folder + 'clean_PTDEMOG.csv')

    demographic_data = demographic_data_1.groupby("RID").first()[
        ['PTGENDER', "PTRACCAT", 'EXAMDATE_bl']].reset_index().merge(
        demographic_data_2.groupby('RID').first()[['PTDOBYY']].reset_index(),
        how='inner', left_on='RID', right_on='RID'
    ).rename(columns={
        'PTGENDER': 'gender',
        'PTRACCAT': 'race',
        'EXAMDATE_bl': 'baseline_exam_date',
        'PTDOBYY': 'year_of_birth'})
    demographic_data['year_of_birth'] = demographic_data['year_of_birth'].astype(int)

    return demographic_data
