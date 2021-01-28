<h3>Skripte zum Extrahieren von Gelenk-Keypoints in verschiedene Formate und aus verschiedenen Modellen heraus.</h3>

<h2>Zusmmenfassung</h2>
<ul>
<li>
amass_keypoint_extraction:<br>
Extrahieren der relevanten Gelenk KEypoints aus dem Amass Datensatz mithilfe des SMPL-H Modells.
</li>
<li>
tf_pose_keypoint_extraction:<br>
Extrahieren von Gelenk KEypoints aus der Posenerkennung von tf-pose auf Bildern der Realsense Kamera mit verschiedneen Koordinatensystemen in ein csv Dokument.
</li>
<li>
tf_pose_keypoint_extraction_xyz: <br>
Identische Extrahierung mit eingeschränkter Auswahl an KEypoints und in .xy Format
</li>
<li>
training_data_preparation: <br>
Skript zum vorbereiten der Trainingsdaten. Hierbei werden aufgenommene Sequenzen und das entsprechende Modell geladen um Trainingssequenzen von jeweils 15 Frames zu generieren. Diese werden annotiert und im .npy Format abgespeichert.
</li>
</ul>
