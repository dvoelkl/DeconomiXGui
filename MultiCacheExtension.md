Hier eine ausführliche, praxisnahe Beschreibung, wie Multi-Session-Support in einem Dash-Projekt wie DeconomiX GUI implementiert werden kann – inklusive Design, Funktionalitäten, Herausforderungen und Best Practices:

0. Hintergrundinformationen zu DeconomiX GUI
DeconomiX GUI ist eine graphische Benutzeroberfläche für das deconomix python package (basierrend auf folgendem Preprint: https://www.biorxiv.org/content/10.1101/2024.11.28.625894v1, Gitlab Implementation: https://gitlab.gwdg.de/MedBioinf/MedicalDataScience/DeconomiX), welches für Dekonvolution von Bulk Transcriptomic Daten verwendet werden kann. Da allerdings viele Nutzer keine Programmierkenntnisse besitzen, ist es notwendig, dass diese eine einfach zu bedienende, intuitive Lösung haben um Transcriptomdaten zu verarbeiten. Die Nutzerbasis besteht aus Biologen, Bioinformatikern und Medizinern aus der ganzen Welt, daher ist die Sprache des Programms auch Englisch. Sämtliche Texte und Codekommentare sollen auf Englisch verfasst sein. Die Plattform kann entweder auf einem Server gehostet werden, so dass sich mehrere Nutzer darauf verbinden können, oder als lokales Programm genutzt werden. Bislang ist der Serversupport allerdings durch den lokalen Cache eingeschränkt, der es nur erlaubt an einem Projekt zu arbeiten, auf das jeder Nutzer Zugriff hat.

Der Workflow läuft wie folgt ab:
-> 1. Der Nutzer selektiert über die Upload Oberfläche entweder eine proprietäre DeconomiX (.dcx) Datei oder import diese über den AnnData (.h5ad) Import
-> 2. Der Nutzer wechselt auf die DTD Seite, wählt seine gewünschten Parameter und führt DTD aus. Die Ergebnisse werden unterhalb visualisiert
-> 3. Nachdem DTD ausgeführt wurde, steht es dem Nutzer frei, einen komplexeren Dekonvolutionsalgorithmus namens ADTD mit wesentlich mehr Parametern zu verwenden. Dazu kann er entweder zuerst einen Hyperparametersearch durchführen, oder ADTD mit gewählten Parametern auf einem ausgewählten Teil des geladenen Datensatzes ausführen
-> 4. Weitere Dekonvolutionsalgorithmen sind in Entwicklung, welche über das Plugin System in Zukunft eingefügt werden sollen
-> 5. Zudem ist geplant, weitere Plugins, wie Survival Analysis oder Genanalysen oder Analysen auf hochregulierte Markergene in ADTD und verknüpfung mit bekanntem Wissen aus Plattformen wie Cellmarker2.0, einzubauen, sollten Patientendaten gegeben sein

1. Konzept: Was ist Multi-Session-Support?
Multi-Session-Support bedeutet, dass mehrere Nutzer (oder ein Nutzer mit mehreren Projekten) unabhängig voneinander mit eigenen Daten, Einstellungen und Analysen arbeiten können. Jede Session ist dabei ein isolierter Arbeitsbereich mit eigenem Status und eigenem Cache.

2. Implementierungsschritte für ein neues Projekt
a) Session-Identifikation und -Verwaltung
Session-ID:
Jede Session erhält eine eindeutige ID (z.B. Zeitstempel + UUID).
Beispiel: 20250520#191019#07c2e7ed

Session-Store:
Die aktuelle Session-ID wird in einem Dash-Store (dcc.Store) gehalten und bei jedem Callback mitgegeben.

Session-Manager:
Eine zentrale Klasse (z.B. SessionCacheManager) verwaltet alle aktiven Sessions und deren Caches.

b) Per-Session-Cache
Cache-Struktur:
Statt eines globalen Caches gibt es für jede Session ein eigenes Cache-Objekt (z.B. DCXCache).

Zugriff:
Alle Plug-ins und Callbacks greifen über eine Hilfsfunktion wie get_session_cache(session_id) auf den Cache der aktuellen Session zu.

Persistenz:
Jeder Session-Cache wird als Datei (z.B. Pickle) gespeichert und beim Start geladen.

c) Session-Operationen
Erstellen:
Neue Session mit eindeutiger ID und leerem Cache anlegen.

Wechseln:
Session-ID im Store aktualisieren, UI und Cache entsprechend umschalten.

Löschen/Archivieren:
Session aus dem Manager entfernen und ggf. auf Festplatte archivieren.

Wiederherstellen:
Archivierte Session wieder in die aktiven Sessions aufnehmen.

3. Design der MultiSession-Page
a) UI-Elemente
Session-Liste:
Tabelle mit allen aktiven und archivierten Sessions, inkl. Name, Erstellungsdatum, Dateiname, Status (z.B. DTD/ADTD durchgeführt).

Aktionen:

Neue Session anlegen
Session wechseln
Session löschen
Session archivieren/wiederherstellen
Session umbenennen
Session exportieren/importieren (im Sinne eines Archivs)
Status-Anzeige:
Zeigt an, welche Session aktiv ist und ob eine Datei geladen wurde.

b) UX-Aspekte
Bestätigung bei kritischen Aktionen (z.B. Löschen)
Schutz der Default-Session (kann nicht gelöscht/archiviert werden)
Feedback bei Fehlern (z.B. Session existiert schon)
4. Funktionalitäten der MultiSession-Page
Session-Übersicht:
Alle Sessions mit Metadaten und Status.

Session-Management:
Erstellen, Wechseln, Löschen, Archivieren, Wiederherstellen.

Synchronisation:
Nach jeder Aktion wird die UI aktualisiert und die aktuelle Session im Store gesetzt.

Session-Status:
Zeigt an, ob eine Datei geladen ist, ob DTD/ADTD durchgeführt wurde etc.

5. Geplante Workflows:
5.1 Normaler DTD/ADTD Workflow
-> 1: Datei über Upload hochladen (Nach dem Hochladen wird eine Session ID erzeugt, die im Session Manager auftaucht)
-> 2: DTD: Parameter wählen, DTD ausführen
-> 3: ADTD: Parameter auswählen / Hyperparameter Search, ADTD ausführen
=> Es sollten nun, wenn die Session wieder im Session Manager ausgewählt wurde, alle vorherigen Ergebnisse wieder angezeigt werden. (Aktuell ist dies mit dem lokalen Cache bereits möglich)

5.2 Neue Session nach DTD/ADTD Workflow
-> [Wie 5.1]
-> 4. Im Session Manager wird die aktuelle Session umbenannt, z.b. auf "melanoma run 1"
-> 5. Eine neue Session wird erzeugt
-> 6. DTD und ADTD Tab werden wieder deaktiviert
-> 7. Es muss wieder eine neue dcx Datei geladen werden
=> Nach dem Laden wieder 5.1

5.3 Session wechseln
-> [Entweder 5.1 oder 5.2]
-> 1. Im Session Manager wird eine beliebige Session ausgewählt
-> 2. Alle vorher aktivierten Tabs werden wieder aktiviert und mit den vorherigen Ergebnissen und gewählten Parametern gefüllt

5.4 Session löschen
-> 1. Im Session Manager wird eine beliebige Session gelöscht
-> 2. Vor dem Entgültigen Löschen erscheint eine Notification, z.b. "Do you really want to peramently delete <SessionName>? Y/N"

5.5 Session archivieren
-> 1. Im Session Manager wird eine beliebige Session archiviert
-> 2. Diese wird aus der Liste von aktiven Session entfernt und in einer extra Liste angezeigt
-> 3. Anstelle von die Session im RAM zu speichern, wird diese nun in einem Ordner auf dem Server gespeichert

5.6 Session re-archivieren
-> 1. Im Session Manager wird eine beliebige Session im Archiv ausgewählt und reaktiviert
-> 2. Die Session wird wieder in den Arbeitsspeicher geladen und die Datei aus dem Ordner auf dem Server entfernt
-> [Danach wie 5.3]

6. Herausforderungen & Stolpersteine
a) Session-Konsistenz
Problem:
Wenn die Session-ID nicht konsistent weitergegeben wird, landen Daten im falschen Cache.

Lösung:
Immer die aktuelle Session-ID aus dem Store an alle Callbacks und Plug-ins weitergeben.

b) Persistenz & Parallelität
Problem:
Bei mehreren Nutzern/Prozessen können Race Conditions oder Inkonsistenzen auftreten.

Lösung:
Zugriff auf den Session-Cache mit Locks schützen, Caches nach jeder Änderung speichern.

c) Navigation & Berechtigungen
Problem:
Nutzer könnten auf Seiten zugreifen, für die die Session nicht bereit ist (z.B. DTD ohne Datei).

Lösung:
Navigation dynamisch aktivieren/deaktivieren, je nach Session-Status.

d) Ressourcenmanagement
Problem:
Alte/ungenutzte Sessions und Caches können Speicherplatz belegen.

Lösung:
Auto-Cleanup-Logik für alte Sessions implementieren (z.B. nach Zeit oder Anzahl, dies soll in der Config Datei sowohl aktiviert oder deaktiviert werden können, als auch die spezifischen Parameter gewählt werden können, z.b. Anzahl der Dateien ab wann gelöscht werden soll oder letzter Zugriff auf die Session ab wann sie gelöscht werden darf).

e) Kompatibilität mit bestehenden Plug-ins
Problem:
Plug-ins, die bisher nur mit einem globalen Cache arbeiten, müssen refaktoriert werden.

Lösung:
Alle Plug-ins so umbauen, dass sie immer die Session-ID und den zugehörigen Cache verwenden.

7. Best Practices
Session-Kontext immer explizit weitergeben (nie auf globale Variablen verlassen)
Korrekte Groß- und Kleinschreibung der Variablennamen beachten
Alle Session-Operationen zentral im Session-Manager kapseln
UI-Feedback für alle Session-Aktionen
Debug-Logging für Session- und Cache-Operationen
Testfälle für Session-Wechsel, parallele Sessions und Fehlerfälle
Sollte Multi-Session-Support entweder nicht als plugin im pages Ordner enthalten sein, oder multi session support in der Konfigurationsdatei von MultiSession deaktiviert sein, so soll das bisherige Cacheverhalten weiterhin verwendet werden.

8. Perspektiven
Perspektivisch sollen Sessions verschlüsselt werden können, sodass nur Nutzer, die das dazugehörige Passwort haben, die Session aus dem Session Viewer laden und einsehen können. Somit soll der korrekte Umgang mit Patientendaten eingehalten werden.

Desweiteren soll der Zugriff von einem Plugin, wie z.B. DTD oder ADTD auf die aktive Session einfach gestaltet werden, sodass eine leichte Erweiterbarkeit des Programms gegeben ist und erfahrene Nutzer sich eigene Erweiterungen für die DeconomiX GUI Plattform leicht erstellen und in bestehende Workflows integrieren können.

Zudem kann in weiter Zukunft für die Serverlösung von DeconomiX GUI über ein Nutzerbasiertes System nachgedacht werden, welches verschiedene Nutzergruppen mit verschiedenen Berechtigungen enthält, sodass manche Nutzer nur auf ihre eigenen Sessions oder die einer Gruppe zugewiesenen Session zugreifen können.

9. Debugging
Zur Erleichterten Fehlersuche, soll eine Debugging Print Option geschaffen werden, welche durch das .config File aktiviert werden kann und sämtliche Änderungen des aktiven Caches in die Konsole schreibt.

10. Zusammenfassung
Multi-Session-Support macht eine Dash-App wie DeconomiX GUI deutlich flexibler und robuster, ist aber mit erhöhtem Aufwand für Session-Management, Persistenz und UI-Logik verbunden. Die größte Herausforderung ist die konsequente Trennung und Weitergabe des Session-Kontexts in allen Teilen der App. Ein gutes Design der MultiSession-Page und ein sauberer, thread-sicherer Session-Manager sind der Schlüssel zu einer stabilen und benutzerfreundlichen Lösung.





KONZEPT:

Multi-Session Architecture & Migration Concept for DeconomiX GUI
1. Overview
The goal is to enable true multi-session support in DeconomiX GUI, so that multiple users (or projects) can work independently with their own data, settings, and results. Each session will have its own isolated cache and persistent state. All plugins and workflows must be migrated to use this new session context.

2. Core Components
2.1 Session ID & Store
Each session is identified by a unique session ID (e.g., timestamp + UUID).
The current session ID is stored in a Dash dcc.Store and passed to all callbacks and plugins.
2.2 SessionCacheManager
A new central class, e.g., SessionCacheManager, manages all active and archived sessions.
Responsibilities:
Create, load, save, delete, archive, and restore session caches.
Maintain a mapping: session_id → DCXCache instance.
Handle persistence (Pickle files in a configurable directory, default: cache).
Provide thread-safe access (with locks if needed).
Optionally, handle auto-cleanup of old/unused sessions.
2.3 Per-Session Cache
Instead of a global localDCXCache, each session gets its own DCXCache object.
All data (uploaded files, results, parameters) are stored per session.
Plugins and callbacks must always access the cache via a helper:
get_session_cache(session_id).
2.4 MultiSession Page
New UI page for session management:
List all active and archived sessions (with metadata: name, creation date, file, status).
Actions: create, switch, rename, delete, archive, restore, export/import.
Status display: which session is active, what data/analyses are present.
Confirmation dialogs for destructive actions.
Feedback for errors (e.g., duplicate session name).
2.5 Debug Logging
All cache accesses (read, write, delete) are logged if enabled in the config.
Logging can be toggled via the config file.
3. Migration Steps
3.1 Core Refactoring
Remove all direct usage of the global localDCXCache.
Introduce a helper function (e.g., get_session_cache(session_id)) to retrieve the correct cache for the current session.
Refactor all plugins and callbacks to require the session ID as an argument (from Dash Store or callback context).
3.2 Session Manager Implementation
Implement SessionCacheManager in utils/session_cache_manager.py.
Methods:
create_session()
get_session(session_id)
save_session(session_id)
delete_session(session_id)
archive_session(session_id)
restore_session(session_id)
list_sessions()
list_archived_sessions()
rename_session(session_id, new_name)
export_session(session_id)
import_session(file_path)
cleanup_sessions() (if enabled)
3.3 UI/UX
Add a new MultiSession page (plugin) for session management.
Update navigation to show the current session and allow switching.
Ensure all actions update the UI and session store accordingly.
3.4 Plugin Migration
Update all plugins (Uploading, DTD, ADTD, etc.) to:
Accept the session ID as input (from Dash Store).
Use get_session_cache(session_id) for all data access.
Update all callbacks to pass the session ID.
Remove any remaining global cache usage.
3.5 Backward Compatibility
If Multi-Session is disabled (via config or missing plugin), fall back to the old global cache behavior.
3.6 Persistence & Cleanup
On every cache change, save the session cache to disk (Pickle).
Implement auto-cleanup logic (configurable, no defaults).
Archived sessions are moved to a separate folder and can be restored.
3.7 Testing
Test all workflows: create, switch, delete, archive, restore, export/import sessions.
Test parallel sessions and error cases.
Test debug logging and UI feedback.
4. Extensibility & Future-Proofing
Design all session/context APIs to be easily extendable (e.g., for encryption, user management).
Keep all session operations encapsulated in the Session Manager.
Ensure that adding user/group-based access in the future will require minimal changes.
5. File/Folder Structure (Proposal)
6. To-Do List
<input disabled="" type="checkbox"> Implement SessionCacheManager and helper functions.
<input disabled="" type="checkbox"> Refactor all plugins and callbacks to use session context.
<input disabled="" type="checkbox"> Implement the MultiSession management page.
<input disabled="" type="checkbox"> Add debug logging for all cache operations.
<input disabled="" type="checkbox"> Update documentation and README.
<input disabled="" type="checkbox"> Provide migration scripts or instructions for existing projects.
<input disabled="" type="checkbox"> Test all workflows and edge cases.