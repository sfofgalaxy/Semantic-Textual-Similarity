import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.nlp.v20190408 import nlp_client, models
try:
    cred = credential.Credential("AKIDQDOKzeaEgqjLgaxAdryp2zh8IBm0KZVC", "YpGPN5WkiOnzwGLAl6F9ZjdGdANTIqTp")
    httpProfile = HttpProfile()
    httpProfile.endpoint = "nlp.tencentcloudapi.com"

    clientProfile = ClientProfile()
    clientProfile.httpProfile = httpProfile
    client = nlp_client.NlpClient(cred, "ap-guangzhou", clientProfile)

    # req = models.TextSimilarityRequest()

    # socres = []
    df = pd.read_csv("./data/test.csv")
    labels = df['label'].tolist()
    # for index, row in tqdm(df.iterrows()):
    #     params = {
    #         "SrcText": row['sentence1'],
    #         "TargetText": [ row['sentence2'] ]
    #     }
    #     req.from_json_string(json.dumps(params))

    #     resp = client.TextSimilarity(req)
    #     resp = resp.to_json_string()
    #     resp = json.loads(resp)
    #     socres.append(resp['Similarity'][0]['Score'])
    # a=np.array(socres)
    # np.save('socres.npy', a)   # 保存为.npy格式

    a=np.load('socres.npy')
    socres=a.tolist()
    def f1(x):
        return 1 if x>0.88 else 0
    preds = list(map(f1,  socres))
    report = classification_report(labels, preds, digits=4)
    print(report)


except TencentCloudSDKException as err:
    print(err)