# Pod-Profile presets

Vordefinierte Profile geben Pods eine sichere Startstruktur:

- `private`: hohe Sensitivity, Owner-ACL, eigener Namespace
- `hobby`: persönlicher Hobby-Namespace
- `beruf`: Work-Domain, vertrauliche Sensitivity, Team-ACL
- `programmierung`: technische Tags und eigener Cache-Namespace

Profile liefern Defaults; sie ersetzen keine expliziten Provenance-/Lifecycle-Felder. Sensitivity und ACL werden nicht durch Soft-Semantik überschrieben.
