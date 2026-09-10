use std::collections::HashSet;

#[test]
fn minicpm_registry_has_exact_model_card_rows() {
    let raw = include_str!("../config/minicpm5-2b.toml");
    let doc: toml::Value = toml::from_str(raw).unwrap();
    let arr = doc.get("benchmark").unwrap().as_array().unwrap();
    assert_eq!(arr.len(), 34);
    let ids = arr
        .iter()
        .map(|v| v.get("id").unwrap().as_str().unwrap())
        .collect::<HashSet<_>>();
    assert_eq!(ids.len(), 34);
}

#[test]
fn every_category_is_present() {
    let raw = include_str!("../config/minicpm5-2b.toml");
    let doc: toml::Value = toml::from_str(raw).unwrap();
    let arr = doc.get("benchmark").unwrap().as_array().unwrap();
    let cats = arr
        .iter()
        .map(|v| v.get("category").unwrap().as_str().unwrap())
        .collect::<HashSet<_>>();
    for required in [
        "Code Reasoning",
        "Math Reasoning",
        "Instruction Following",
        "General Knowledge",
        "Long Context",
        "Tool Use",
        "Coding Agent",
        "Search Agent",
        "General Agent",
    ] {
        assert!(cats.contains(required), "missing category {required}");
    }
}
