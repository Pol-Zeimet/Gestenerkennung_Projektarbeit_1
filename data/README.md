<h3>Übersicht</h3>
Dieser Ordner enthält extrahierte Gelenkpositionen sowohl für einzelne Posen als auch für ganze Bewegungen.
Daten im .xyz Format können in einem Meshviewer (bsp. Meshlab) angesehen werden.
Um eine Bewegungsabfolge anzuzeigen muss zuerst mit dem mdd Skript eine Sammlung von .xyz Files zu einem mesh.obj und einer mesh.mdd umgewandelt werden. Diese können dann z.bsp in Blender visualisiert werden.

<h3>Aktueller Inhalt</h3>
<ul>
<li>
movement: <br>
Aus dem Amass Datensatz extrahierte Keypoints in verschiedenen Posen und Bewegungen. Trainingsdaten für das Modell zur Bewegungsklassifizierung.
</li>
<li>
recording_samples:<br>
Auszug aus den von Tf-Pose vorhergesagten Keypoints in verschiedenen Posen und Bewegungen.
</li>
</ul>

<h3>Zu den Trainingsdaten</h3>
Die Trainingsdaten liegen aktuell im .npy Format vor und enthalten:
<ul>
<li>
340 Sequenzen von jeweils 15 Frames für sonstige Bewegungen
</li>
<li>
95 Sequenzen von jeweils 15 Frames für Boxbewegungen
</li>
Total: 435 Sequenzen

Es empfielt sich, weitere Trainingsdaten hinzuzufügen.
