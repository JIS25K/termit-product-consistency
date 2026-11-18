"""Source-only configuration, full-image registration, conservative ink transfer.

Does not load evaluation fixtures, deletion masks, original controlled image,
Run 22 target coordinates, per-target boxes, or OCR annotations.
"""
import datetime, hashlib, json, time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
CFG = json.loads((ROOT / 'protocol.json').read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def project(points, H):
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)

def rectangle(roi):
    x0,y0,x1,y1 = roi
    return np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])

def match_filter(a, b, ratio):
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a, b, k=2)
    return {m.queryIdx:m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance < ratio*n.distance}

def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    reference_path = (ROOT / CFG['reference']).resolve()
    reference = cv2.imread(str(reference_path))
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    quad = np.float32(CFG['source_label_quad'])
    roi = CFG['source_text_roi']
    x0,y0,x1,y1 = roi
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    mask[y0-12:y1+12,x0-12:x1+12] = 0
    a = CFG['alignment']
    sift = cv2.SIFT_create(nfeatures=a['features'], contrastThreshold=a['contrast_threshold'])
    key_ref, desc_ref = sift.detectAndCompute(gray, mask)
    alpha = np.zeros(gray.shape, np.float32)
    alpha[y0:y1,x0:x1] = np.clip((170-gray[y0:y1,x0:x1].astype(float))/100, 0, 1)
    cv2.imwrite(str(OUT/'source_feature_mask.png'), mask)
    cv2.imwrite(str(OUT/'source_ink_mask.png'), np.rint(alpha*255).astype(np.uint8))
    records=[]
    for case in CFG['cases']:
        t0 = time.perf_counter()
        input_path = (ROOT/case['input']).resolve()
        before_hash = sha(input_path)
        candidate = cv2.imread(str(input_path))
        h,w = candidate.shape[:2]
        target_gray = cv2.cvtColor(candidate,cv2.COLOR_BGR2GRAY)
        output = candidate.copy()
        support = np.zeros((h,w),bool)
        diagnostic = candidate.copy()
        record = {'id':case['id'], 'role':case['role'], 'input':str(input_path), 'input_sha256':before_hash,
                  'decision':'abstain_alignment', 'reasons':[], 'source_keypoints':len(key_ref)}
        key_target, desc_target = sift.detectAndCompute(target_gray,None)
        matches=[]
        if desc_target is not None and len(desc_target)>=2:
            forward = match_filter(desc_ref,desc_target,a['ratio'])
            reverse = match_filter(desc_target,desc_ref,a['ratio'])
            matches = [m for i,m in forward.items() if m.trainIdx in reverse and reverse[m.trainIdx].trainIdx==i]
        record['mutual_matches']=len(matches)
        H=None
        if len(matches)>=4:
            src=np.float32([key_ref[m.queryIdx].pt for m in matches])
            dst=np.float32([key_target[m.trainIdx].pt for m in matches])
            cv2.setRNGSeed(a['seed'])
            H,inlier_mask=cv2.findHomography(src,dst,cv2.RANSAC,a['ransac_px'],maxIters=a['ransac_iterations'],confidence=.995)
        if H is None:
            record['reasons'].append('homography unavailable')
        else:
            keep=inlier_mask.ravel().astype(bool)
            si,di=src[keep],dst[keep]
            count=len(si)
            error=np.linalg.norm(project(si,H)-di,axis=1)
            q=project(quad,H); name_q=project(rectangle(roi),H)
            source_area=abs(cv2.contourArea(quad))
            hull_frac=abs(cv2.contourArea(cv2.convexHull(si)))/source_area
            span=np.ptp(si,axis=0)/np.ptp(quad,axis=0)
            label_area=abs(cv2.contourArea(q))
            canvas=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
            convex=bool(cv2.isContourConvex(q))
            visible_area=cv2.intersectConvexConvex(q,canvas)[0] if convex else 0
            name_height=float(min(np.linalg.norm(name_q[3]-name_q[0]),np.linalg.norm(name_q[2]-name_q[1])))
            above=int((si[:,1]<y0).sum()); below=int((si[:,1]>y1).sum())
            record.update({'inliers':count,'inlier_ratio':count/len(matches),'median_reprojection_px':float(np.median(error)),
                           'source_hull_fraction':hull_frac,'source_span_xy':span.tolist(),'inliers_above_below_name':[above,below],
                           'H_source_to_target':H.tolist(),'projected_label_quad':q.tolist(),'projected_name_quad':name_q.tolist(),
                           'projected_name_height_px':name_height})
            checks={
                'too few inliers':count>=a['min_inliers'],
                'weak inlier ratio':count/len(matches)>=a['min_inlier_ratio'],
                'large reprojection error':float(np.median(error))<=a['max_median_reprojection_px'],
                'limited spatial support':hull_frac>=a['min_source_hull_fraction'] and span[0]>=a['min_source_x_span'] and span[1]>=a['min_source_y_span'],
                'no support bracketing text':min(above,below)>=a['min_inliers_above_and_below_name'],
                'invalid projected label':convex and cv2.contourArea(q,oriented=True)*cv2.contourArea(quad,oriented=True)>0,
                'implausible label scale':a['min_projected_label_area']<=label_area<=h*w*a['max_projected_label_area_fraction'],
                'label mostly off canvas':visible_area/max(label_area,1)>=a['min_label_visible_fraction'],
                'name off canvas':bool(np.all(name_q>=0) and np.all(name_q[:,0]<w) and np.all(name_q[:,1]<h)),
                'name too small':name_height>=a['min_projected_name_height_px']}
            record['reasons']=[reason for reason,ok in checks.items() if not ok]
            cv2.polylines(diagnostic,[np.rint(q).astype(np.int32)],True,(255,180,0),2)
            cv2.polylines(diagnostic,[np.rint(name_q).astype(np.int32)],True,(0,0,255),2)
            for p in di: cv2.circle(diagnostic,tuple(np.rint(p).astype(int)),3,(0,200,0),-1)
            if not record['reasons']:
                rectified=cv2.warpPerspective(target_gray,np.linalg.inv(H),(gray.shape[1],gray.shape[0]))
                gx0,gy0,gx1,gy1=CFG['presence_guard_roi']
                region=rectified[gy0:gy1,gx0:gx1]
                pc=CFG['presence']
                threshold=min(pc['dark_threshold_cap'],float(np.percentile(region,85))-pc['dark_contrast_from_p85'])
                occupancy=float((region<threshold).mean())
                record.update({'guard_dark_ink_fraction':occupancy,'dark_threshold':threshold})
                if occupancy>=pc['present_min_ink_fraction']:
                    record['decision']='preserve_existing_ink'
                    record['reasons']=['ink present; semantic correctness not assessed']
                elif occupancy>pc['blank_max_ink_fraction']:
                    record['decision']='abstain_ambiguous_ink'
                    record['reasons']=['uncertain blankness']
                else:
                    warped_alpha=cv2.warpPerspective(alpha,H,(w,h),flags=cv2.INTER_LINEAR)
                    warped_color=cv2.warpPerspective(reference,H,(w,h),flags=cv2.INTER_LINEAR)
                    output=np.rint(candidate*(1-warped_alpha[:,:,None])+warped_color*warped_alpha[:,:,None]).clip(0,255).astype(np.uint8)
                    support=warped_alpha>0
                    record['decision']='restored'
        changed=np.any(candidate!=output,axis=2)
        outside=int((changed&~support).sum())
        assert outside==0
        assert sha(input_path)==before_hash
        if record['decision']!='restored': assert not changed.any()
        path=OUT/(case['id']+'.png')
        cv2.imwrite(str(path),output)
        cv2.imwrite(str(OUT/(case['id']+'_support.png')),support.astype(np.uint8)*255)
        cv2.imwrite(str(OUT/(case['id']+'_alignment.jpg')),diagnostic)
        assert np.array_equal(cv2.imread(str(path)),output)
        record.update({'changed_pixels':int(changed.sum()),'support_pixels':int(support.sum()),'changed_outside_support':outside,
                       'output_sha256':sha(path),'elapsed_seconds':time.perf_counter()-t0,'target_manual_alignment_parameters':0})
        records.append(record)
        print(case['id'],record['decision'],record['changed_pixels'],record['reasons'])
    summary={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'protocol_sha256':sha(ROOT/'protocol.json'),'code_sha256':sha(__file__),
             'reference_sha256':sha(reference_path),'elapsed_seconds':time.perf_counter()-started,
             'generative_calls':0,'cases':records}
    (OUT/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
