Skripte von Dritten:
<ul>
<li>mdd:<br>
Zum generieren von .obj und .mdd dokumentsn zum anzeigen von Bewegungen in Blender.<br>
Ausführung:
    $ animate.py filepath/name_of_file_1.xyz filepath/name_of_file_2.xyz filepath/name.obj filepath/name.mdd    
Das Skript nimmt eine Abfolge von xyz Dokumenten entgegen und generiert eine Bewegung daraus.
Namen und Pfade sind individualisierbar, für dateien können wildcards verwendet werden, um mherere Dateien in einem Ordner abzugreifen. Bsp:
    $ animate.py filepath/name_of_file_*.xyz  filepath/name.obj filepath/name.mdd   
oder
    $ animate.py filepath/*.xyz  filepath/name.obj filepath/name.mdd   
</li>
<li>mano_v1_2:<br>
Wird benötigt, um AMASS Körpermodelle zu laden</li>
<li>tf-pose:<br>
Enthält eine estimator.py, welche die Original estimator.py im Tf-pose-estimation Projekt ersetzen soll.</li>
</ul>