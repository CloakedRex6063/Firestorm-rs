use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemStruct, parse_macro_input};

#[allow(non_snake_case)]
#[proc_macro_attribute]
pub fn Handle(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    let name = &input.ident;

    let struct_with_derives = quote! {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #input
    };

    let handle_impl = quote! {
        impl #name {
            pub fn new(id: u32) -> Self {
                #name(id)
            }
        }
    };

    let expanded = quote! {
        #struct_with_derives
        #handle_impl
    };

    TokenStream::from(expanded)
}
